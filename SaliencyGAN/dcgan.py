import os

import torch
import torch.optim as optim
from absl import flags, app
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tensorboardX import SummaryWriter
from tqdm import trange

import source.models.dcgan as models
import source.losses as losses
from source.utils import generate_imgs, infiniteloop, set_seed
from metrics.score.both import get_inception_score_and_fid

import numpy as np
import cv2

net_G_models = {
    'cnn32': models.Generator32,
    'cnn48': models.Generator48,
}

net_D_models = {
    'cnn32': models.Discriminator32,
    'cnn48': models.Discriminator48,
}

loss_fns = {
    'bce': losses.BCEWithLogits,
    'hinge': losses.Hinge,
    'was': losses.Wasserstein,
    'softplus': losses.Softplus
}

FLAGS = flags.FLAGS
# model and training
flags.DEFINE_enum('dataset', 'cifar10', ['cifar10', 'stl10'], "dataset")
flags.DEFINE_enum('arch', 'cnn32', net_G_models.keys(), "architecture")
flags.DEFINE_integer('total_steps', 50000, "total number of training steps")
flags.DEFINE_integer('batch_size', 128, "batch size")
flags.DEFINE_float('lr_G', 2e-4, "Generator learning rate")
flags.DEFINE_float('lr_D', 2e-4, "Discriminator learning rate")
flags.DEFINE_multi_float('betas', [0.5, 0.9], "for Adam")
flags.DEFINE_integer('n_dis', 1, "update Generator every this steps")
flags.DEFINE_integer('z_dim', 100, "latent space dimension")
flags.DEFINE_enum('loss', 'bce', loss_fns.keys(), "loss function")
flags.DEFINE_integer('seed', 0, "random seed")
# logging
flags.DEFINE_integer('eval_step', 5000, "evaluate FID and Inception Score")
flags.DEFINE_integer('sample_step', 500, "sample image every this steps")
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_string('logdir', './logs/DCGAN_CIFAR10', 'logging folder')
flags.DEFINE_bool('record', True, "record inception score and FID score")
flags.DEFINE_string('fid_cache', './stats/cifar10_stats.npz', 'FID cache')
# generate
flags.DEFINE_bool('generate', False, 'generate images')
flags.DEFINE_string('pretrain', None, 'path to test model')
flags.DEFINE_string('output', './outputs', 'path to output dir')
flags.DEFINE_integer('num_images', 50000, 'the number of generated images')

device = torch.device('cuda:0')
ds_step = 30000
alpha = 0.6
def get_saliency(image) :

    imageHeight, imageWidth = image.shape[:2]
    resizeImage = image

    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(resizeImage)
    saliencyMap = (saliencyMap * 255).astype("uint8") 

    threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255,
                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    contours, hierachy = cv2.findContours(threshMap, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    area, output = 0, contours[0]

    for cnt in contours:
        if (area < cv2.contourArea(cnt)):
            area = cv2.contourArea(cnt)
            output = cnt

    epsilon = 0.02 * cv2.arcLength(output, True)
    approx = cv2.approxPolyDP(output, epsilon, True)

    x, y, w, h = cv2.boundingRect(approx)
    x = (x+w)//2
    y = (y+h)//2
    if x-8<0:
        lx = 0
        rx = x+(x-8)
    if x+8 >31: #because cifar10
        rx = 31
        lx = x - (x+8-31)
    if y-8<0:
        ly = 0
        ry = y+(y-8)
    if y+8 >31: #because cifar10
        ry = 31
        ly = y - (y+8-31)
    #사진 자르기
    dst = resizeImage[ly:ry,lx:rx]  # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]]
    return dst

class Discriminator(nn.Module):
    def __init__(self, M):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # 32
            nn.Conv2d(3, 64, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 16
            nn.Conv2d(64, 128, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),
            # 8
            nn.Conv2d(128, 256, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(256),
            # 4
            nn.Conv2d(256, 512, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(512)
            # 1
        )

        self.linear = nn.Linear(M // 16 * M // 16 * 512, 1)

    def forward(self, x):
        x = self.main(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x


def generate():
    assert FLAGS.pretrain is not None, "set model weight by --pretrain [model]"

    net_G = net_G_models[FLAGS.arch](FLAGS.z_dim).to(device)
    net_G.load_state_dict(torch.load(FLAGS.pretrain)['net_G'])
    net_G.eval()

    counter = 0
    os.makedirs(FLAGS.output)
    with torch.no_grad():
        for start in trange(
                0, FLAGS.num_images, FLAGS.batch_size, dynamic_ncols=True):
            batch_size = min(FLAGS.batch_size, FLAGS.num_images - start)
            z = torch.randn(batch_size, FLAGS.z_dim).to(device)
            x = net_G(z).cpu()
            x = (x + 1) / 2
            for image in x:
                save_image(
                    image, os.path.join(FLAGS.output, '%d.png' % counter))
                counter += 1


def train():
    if FLAGS.dataset == 'cifar10':
        dataset = datasets.CIFAR10(
            './data', train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    if FLAGS.dataset == 'stl10':
        dataset = datasets.STL10(
            './data', split='unlabeled', download=True,
            transform=transforms.Compose([
                transforms.Resize((48, 48)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=4,
        drop_last=True)

    net_G = net_G_models[FLAGS.arch](FLAGS.z_dim).to(device)
    net_D = net_D_models[FLAGS.arch]().to(device)
    net_DS = Discriminator(16)

    loss_fn = loss_fns[FLAGS.loss]()

    optim_G = optim.Adam(net_G.parameters(), lr=FLAGS.lr_G, betas=FLAGS.betas)
    optim_D = optim.Adam(net_D.parameters(), lr=FLAGS.lr_D, betas=FLAGS.betas)
    optim_DS = optim.Adam(net_DS.parameters(),lr=FLAGS.lr_D,betas=FLAGS.betas)
    sched_G = optim.lr_scheduler.LambdaLR(
        optim_G, lambda step: 1 - step / FLAGS.total_steps)
    sched_D = optim.lr_scheduler.LambdaLR(
        optim_D, lambda step: 1 - step / FLAGS.total_steps)
    sched_DS = optim.lr_scheduler.LambdaLR(
        optim_DS,lambda step: 1 - step / FLAGS.total_Steps)
    
    os.makedirs(os.path.join(FLAGS.logdir, 'sample'))
    writer = SummaryWriter(os.path.join(FLAGS.logdir))
    sample_z = torch.randn(FLAGS.sample_size, FLAGS.z_dim).to(device)
    with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
        f.write(FLAGS.flags_into_string())
    writer.add_text(
        "flagfile", FLAGS.flags_into_string().replace('\n', '  \n'))

    real, _ = next(iter(dataloader))
    grid = (make_grid(real[:FLAGS.sample_size]) + 1) / 2
    writer.add_image('real_sample', grid)

    looper = infiniteloop(dataloader)
    with trange(1, FLAGS.total_steps + 1, desc='Training', ncols=0) as pbar:
        for step in pbar:
            # Discriminator
            if step == ds_step:
                for _ in ds_step:    
                    for _ in range(FLAGS.n_dis):
                        with torch.no_grad():
                            z = torch.randn(FLAGS.batch_size, FLAGS.z_dim).to(device)
                            fake = net_G(z).detach()
                            fake = get_saliency(fake)
                        real = next(looper).to(device)
                        real = get_saliency(real)
                        net_DS_real = net_DS(real)
                        net_DS_fake = net_DS(fake)
                        loss = loss_fn(net_DS_real, net_DS_fake)

                        optim_DS.zero_grad()
                        loss.backward()
                        optim_DS.step()
            if step > ds_step:
                for _ in range(FLAGS.n_dis):
                    with torch.no_grad():
                        z = torch.randn(FLAGS.batch_size, FLAGS.z_dim).to(device)
                        fake = net_G(z).detach()
                        sfake = get_saliency(fake)
                    real = next(looper).to(device)
                    sreal = get_saliency(real)
                    net_D_real = net_D(real)
                    net_D_fake = net_D(fake)
                    net_DS_real=net_DS(sreal)
                    net_DS_fake = net_DS(sfake)
                    loss = loss_fn(net_D_real, net_D_fake)
                    loss_s =  + loss_fn(net_DS_real,net_DS_fake)
                optim_D.zero_grad()
                optim_DS.zero_grad()
                loss.backward()
                loss_s.backward()
                optim_D.step()
                optim_DS.step()
            else:
                for _ in range(FLAGS.n_dis):
                    with torch.no_grad():
                        z = torch.randn(FLAGS.batch_size, FLAGS.z_dim).to(device)
                        fake = net_G(z).detach()
                    real = next(looper).to(device)
                    net_D_real = net_D(real)
                    net_D_fake = net_D(fake)

                    optim_D.zero_grad()

                    loss.backward()
                    optim_D.step()

                    if FLAGS.loss == 'was':
                        loss = -loss
                    pbar.set_postfix(loss='%.4f' % loss)
            writer.add_scalar('loss', loss, step)

            # Generator
            z = torch.randn(FLAGS.batch_size * 2, FLAGS.z_dim).to(device)
            if step < ds_step:
                loss = loss_fn(net_D(net_G(z)))
            else:
                loss = loss_fn(net_D(net_G(z))*alpha + net_DS(get_saliency(net_G(z))))*(1-alpha)
            optim_G.zero_grad()
            loss.backward()
            optim_G.step()

            sched_G.step()
            sched_D.step()
            sched_DS.step()
            pbar.update(1)

            if step == 1 or step % FLAGS.sample_step == 0:
                fake = net_G(sample_z).cpu()
                grid = (make_grid(fake) + 1) / 2
                writer.add_image('sample', grid, step)
                save_image(grid, os.path.join(
                    FLAGS.logdir, 'sample', '%d.png' % step))

            if step == 1 or step % FLAGS.eval_step == 0:
                torch.save({
                    'net_G': net_G.state_dict(),
                    'net_D': net_D.state_dict(),
                    'optim_G': optim_G.state_dict(),
                    'optim_D': optim_D.state_dict(),
                    'sched_G': sched_G.state_dict(),
                    'sched_D': sched_D.state_dict(),
                }, os.path.join(FLAGS.logdir, 'model.pt'))
                if FLAGS.record:
                    imgs = generate_imgs(
                        net_G, device, FLAGS.z_dim, 50000, FLAGS.batch_size)
                    IS, FID = get_inception_score_and_fid(
                        imgs, FLAGS.fid_cache, verbose=True)
                    pbar.write(
                        "%s/%s Inception Score: %.3f(%.5f), "
                        "FID Score: %6.3f" % (
                            step, FLAGS.total_steps, IS[0], IS[1],
                            FID))
                    writer.add_scalar('Inception_Score', IS[0], step)
                    writer.add_scalar('Inception_Score_std', IS[1], step)
                    writer.add_scalar('FID', FID, step)
    writer.close()


def main(argv):
    set_seed(FLAGS.seed)
    if FLAGS.generate:
        generate()
    else:
        train()


if __name__ == '__main__':
    app.run(main)
