import time, itertools
from dataset import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from networks import *
from utils import *
from utils import ResizePad
from torch.cuda.amp import autocast, GradScaler
import torch.utils.checkpoint as checkpoint
from glob import glob


class UGATIT(object) :
    def __init__(self, args):
        self.light = args.light

        if self.light :
            self.model_name = 'UGATIT_light'
        else :
            self.model_name = 'UGATIT'

        self.result_dir = args.result_dir
        self.dataset = args.dataset

        self.iteration = args.iteration
        self.decay_flag = args.decay_flag

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.ch = args.ch

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.cam_weight = args.cam_weight
        self.global_dis_ratio = args.global_dis_ratio
        self.local_dis_ratio = 1.0 - self.global_dis_ratio

        self.use_ds = args.use_ds
        self.style_dim = args.style_dim
        self.ds_weight = args.ds_weight

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis

        self.img_size = args.img_size
        self.aspect_ratio = args.aspect_ratio
        self.img_w = int(self.img_size * self.aspect_ratio)
        self.img_ch = args.img_ch

        self.device = args.device
        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume
        self.resume_iter = args.resume_iter
        self.amp = args.amp
        self.use_checkpoint = args.use_checkpoint

        if torch.backends.cudnn.enabled and self.benchmark_flag:
            print('set benchmark !')
            torch.backends.cudnn.benchmark = True

        print()

        print("##### Information #####")
        print("# light : ", self.light)
        print("# dataset : ", self.dataset)
        print("# batch_size : ", self.batch_size)
        print("# iteration per epoch : ", self.iteration)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### Discriminator #####")
        print("# discriminator layer : ", self.n_dis)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# cycle_weight : ", self.cycle_weight)
        print("# identity_weight : ", self.identity_weight)
        print("# cam_weight : ", self.cam_weight)
        print("# global_dis_ratio : ", self.global_dis_ratio)
        print("# local_dis_ratio : ", self.local_dis_ratio)
        print("# use_ds : ", self.use_ds)
        print("# style_dim : ", self.style_dim)
        print("# ds_weight : ", self.ds_weight)

        if self.amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None

    def _call(self, module, *args):
        if self.use_checkpoint:
            return checkpoint.checkpoint(lambda *inputs: module(*inputs), *args)
        return module(*args)

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ DataLoader """
        train_transform = transforms.Compose([
            ResizePad((self.img_size, self.img_w)),
            transforms.RandomHorizontalFlip(),
            # transforms.Resize((self.img_size + 30, self.img_size+30)),
            # transforms.RandomCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        test_transform = transforms.Compose([
            ResizePad((self.img_size, self.img_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.trainA = ImageFolder(os.path.join('dataset', self.dataset, 'trainA'), train_transform)
        self.trainB = ImageFolder(os.path.join('dataset', self.dataset, 'trainB'), train_transform)
        self.testA = ImageFolder(os.path.join('dataset', self.dataset, 'testA'), test_transform)
        self.testB = ImageFolder(os.path.join('dataset', self.dataset, 'testB'), test_transform)
        self.trainA_loader = DataLoader(self.trainA, batch_size=self.batch_size, shuffle=True)
        self.trainB_loader = DataLoader(self.trainB, batch_size=self.batch_size, shuffle=True)
        self.testA_loader = DataLoader(self.testA, batch_size=1, shuffle=False)
        self.testB_loader = DataLoader(self.testB, batch_size=1, shuffle=False)

        """ Define Generator, Discriminator """
        self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res,
                                      img_height=self.img_size, img_width=self.img_w,
                                      light=self.light, style_dim=self.style_dim,
                                      use_ds=self.use_ds).to(self.device)
        self.genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res,
                                      img_height=self.img_size, img_width=self.img_w,
                                      light=self.light, style_dim=self.style_dim,
                                      use_ds=self.use_ds).to(self.device)
        self.disGA = Discriminator(input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
        self.disGB = Discriminator(input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
        self.disLA = Discriminator(input_nc=3, ndf=self.ch, n_layers=5).to(self.device)
        self.disLB = Discriminator(input_nc=3, ndf=self.ch, n_layers=5).to(self.device)

        """ Define Loss """
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)

        """ Trainer """
        self.G_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.D_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(), self.disGB.parameters(), self.disLA.parameters(), self.disLB.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)

        """ Define Rho clipper to constraint the value of rho in AdaILN and ILN"""
        self.Rho_clipper = RhoClipper(0, 1)

    def train(self):
        self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()

        start_iter = 1
        if self.resume:
            model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pt'))
            if not len(model_list) == 0:
                model_list.sort()
                start_iter = int(model_list[-1].split('_')[-1].split('.')[0])
                self.load(os.path.join(self.result_dir, self.dataset, 'model'), start_iter)
                print(" [*] Load SUCCESS")
                if self.decay_flag and start_iter > (self.iteration // 2):
                    self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)
                    self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)

        # training loop
        print('training start !')
        start_time = time.time()
        trainA_iter = iter(self.trainA_loader)
        trainB_iter = iter(self.trainB_loader)
        testA_iter = iter(self.testA_loader)
        testB_iter = iter(self.testB_loader)
        for step in range(start_iter, self.iteration + 1):
            if self.decay_flag and step > (self.iteration // 2):
                self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
                self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))

            try:
                real_A, _ = trainA_iter.next()
            except:
                trainA_iter = iter(self.trainA_loader)
                real_A, _ = trainA_iter.next()

            try:
                real_B, _ = trainB_iter.next()
            except:
                trainB_iter = iter(self.trainB_loader)
                real_B, _ = trainB_iter.next()

            real_A, real_B = real_A.to(self.device), real_B.to(self.device)

            # Update D
            self.D_optim.zero_grad()
            with autocast(enabled=self.amp):
                fake_A2B, _, _ = self._call(self.genA2B, real_A)
                fake_B2A, _, _ = self._call(self.genB2A, real_B)

                real_GA_logit, real_GA_cam_logit, _ = self._call(self.disGA, real_A)
                real_LA_logit, real_LA_cam_logit, _ = self._call(self.disLA, real_A)
                real_GB_logit, real_GB_cam_logit, _ = self._call(self.disGB, real_B)
                real_LB_logit, real_LB_cam_logit, _ = self._call(self.disLB, real_B)

                fake_GA_logit, fake_GA_cam_logit, _ = self._call(self.disGA, fake_B2A)
                fake_LA_logit, fake_LA_cam_logit, _ = self._call(self.disLA, fake_B2A)
                fake_GB_logit, fake_GB_cam_logit, _ = self._call(self.disGB, fake_A2B)
                fake_LB_logit, fake_LB_cam_logit, _ = self._call(self.disLB, fake_A2B)

            D_ad_loss_GA = self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).to(self.device)) + self.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.device))
            D_ad_cam_loss_GA = self.MSE_loss(real_GA_cam_logit, torch.ones_like(real_GA_cam_logit).to(self.device)) + self.MSE_loss(fake_GA_cam_logit, torch.zeros_like(fake_GA_cam_logit).to(self.device))
            D_ad_loss_LA = self.MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit).to(self.device)) + self.MSE_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit).to(self.device))
            D_ad_cam_loss_LA = self.MSE_loss(real_LA_cam_logit, torch.ones_like(real_LA_cam_logit).to(self.device)) + self.MSE_loss(fake_LA_cam_logit, torch.zeros_like(fake_LA_cam_logit).to(self.device))
            D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).to(self.device)) + self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device))
            D_ad_cam_loss_GB = self.MSE_loss(real_GB_cam_logit, torch.ones_like(real_GB_cam_logit).to(self.device)) + self.MSE_loss(fake_GB_cam_logit, torch.zeros_like(fake_GB_cam_logit).to(self.device))
            D_ad_loss_LB = self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit).to(self.device)) + self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit).to(self.device))
            D_ad_cam_loss_LB = self.MSE_loss(real_LB_cam_logit, torch.ones_like(real_LB_cam_logit).to(self.device)) + self.MSE_loss(fake_LB_cam_logit, torch.zeros_like(fake_LB_cam_logit).to(self.device))

            D_loss_A = self.adv_weight * (
                self.global_dis_ratio * (D_ad_loss_GA + D_ad_cam_loss_GA) +
                self.local_dis_ratio * (D_ad_loss_LA + D_ad_cam_loss_LA))
            D_loss_B = self.adv_weight * (
                self.global_dis_ratio * (D_ad_loss_GB + D_ad_cam_loss_GB) +
                self.local_dis_ratio * (D_ad_loss_LB + D_ad_cam_loss_LB))

            Discriminator_loss = D_loss_A + D_loss_B
            if self.amp:
                self.scaler.scale(Discriminator_loss).backward()
                self.scaler.step(self.D_optim)
                self.scaler.update()
            else:
                Discriminator_loss.backward()
                self.D_optim.step()

            # Update G
            self.G_optim.zero_grad()
            with autocast(enabled=self.amp):
                fake_A2B, fake_A2B_cam_logit, _ = self._call(self.genA2B, real_A)
                fake_B2A, fake_B2A_cam_logit, _ = self._call(self.genB2A, real_B)

                fake_A2B2A, _, _ = self._call(self.genB2A, fake_A2B)
                fake_B2A2B, _, _ = self._call(self.genA2B, fake_B2A)

                fake_A2A, fake_A2A_cam_logit, _ = self._call(self.genB2A, real_A)
                fake_B2B, fake_B2B_cam_logit, _ = self._call(self.genA2B, real_B)

                fake_GA_logit, fake_GA_cam_logit, _ = self._call(self.disGA, fake_B2A)
                fake_LA_logit, fake_LA_cam_logit, _ = self._call(self.disLA, fake_B2A)
                fake_GB_logit, fake_GB_cam_logit, _ = self._call(self.disGB, fake_A2B)
                fake_LB_logit, fake_LB_cam_logit, _ = self._call(self.disLB, fake_A2B)

            G_ad_loss_GA = self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device))
            G_ad_cam_loss_GA = self.MSE_loss(fake_GA_cam_logit, torch.ones_like(fake_GA_cam_logit).to(self.device))
            G_ad_loss_LA = self.MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit).to(self.device))
            G_ad_cam_loss_LA = self.MSE_loss(fake_LA_cam_logit, torch.ones_like(fake_LA_cam_logit).to(self.device))
            G_ad_loss_GB = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device))
            G_ad_cam_loss_GB = self.MSE_loss(fake_GB_cam_logit, torch.ones_like(fake_GB_cam_logit).to(self.device))
            G_ad_loss_LB = self.MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit).to(self.device))
            G_ad_cam_loss_LB = self.MSE_loss(fake_LB_cam_logit, torch.ones_like(fake_LB_cam_logit).to(self.device))

            G_recon_loss_A = self.L1_loss(fake_A2B2A, real_A)
            G_recon_loss_B = self.L1_loss(fake_B2A2B, real_B)

            G_identity_loss_A = self.L1_loss(fake_A2A, real_A)
            G_identity_loss_B = self.L1_loss(fake_B2B, real_B)

            G_cam_loss_A = self.BCE_loss(fake_B2A_cam_logit, torch.ones_like(fake_B2A_cam_logit).to(self.device)) + self.BCE_loss(fake_A2A_cam_logit, torch.zeros_like(fake_A2A_cam_logit).to(self.device))
            G_cam_loss_B = self.BCE_loss(fake_A2B_cam_logit, torch.ones_like(fake_A2B_cam_logit).to(self.device)) + self.BCE_loss(fake_B2B_cam_logit, torch.zeros_like(fake_B2B_cam_logit).to(self.device))

            if self.use_ds and self.ds_weight > 0:
                # Diversity sensitive loss
                z_a1 = torch.randn(real_A.size(0), self.style_dim, device=self.device)
                z_a2 = torch.randn(real_A.size(0), self.style_dim, device=self.device)
                ds_A1, _, _ = self._call(self.genA2B, real_A, z_a1)
                ds_A2, _, _ = self._call(self.genA2B, real_A, z_a2)
                z_b1 = torch.randn(real_B.size(0), self.style_dim, device=self.device)
                z_b2 = torch.randn(real_B.size(0), self.style_dim, device=self.device)
                ds_B1, _, _ = self._call(self.genB2A, real_B, z_b1)
                ds_B2, _, _ = self._call(self.genB2A, real_B, z_b2)
                DS_loss = -(self.L1_loss(ds_A1, ds_A2.detach()) + self.L1_loss(ds_B1, ds_B2.detach()))
            else:
                DS_loss = torch.tensor(0.0, device=self.device)

            G_loss_A = self.adv_weight * (
                self.global_dis_ratio * (G_ad_loss_GA + G_ad_cam_loss_GA) +
                self.local_dis_ratio * (G_ad_loss_LA + G_ad_cam_loss_LA)) \
                + self.cycle_weight * G_recon_loss_A \
                + self.identity_weight * G_identity_loss_A \
                + self.cam_weight * G_cam_loss_A
            G_loss_B = self.adv_weight * (
                self.global_dis_ratio * (G_ad_loss_GB + G_ad_cam_loss_GB) +
                self.local_dis_ratio * (G_ad_loss_LB + G_ad_cam_loss_LB)) \
                + self.cycle_weight * G_recon_loss_B \
                + self.identity_weight * G_identity_loss_B \
                + self.cam_weight * G_cam_loss_B

            Generator_loss = G_loss_A + G_loss_B + self.ds_weight * DS_loss
            if self.amp:
                self.scaler.scale(Generator_loss).backward()
                self.scaler.step(self.G_optim)
                self.scaler.update()
            else:
                Generator_loss.backward()
                self.G_optim.step()

            # clip parameter of AdaILN and ILN, applied after optimizer step
            self.genA2B.apply(self.Rho_clipper)
            self.genB2A.apply(self.Rho_clipper)

            print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (step, self.iteration, time.time() - start_time, Discriminator_loss, Generator_loss))
            if step % self.print_freq == 0:
                train_sample_num = 5
                test_sample_num = 5
                A2B = np.zeros((self.img_size * 7, 0, 3))
                B2A = np.zeros((self.img_size * 7, 0, 3))

                self.genA2B.eval(), self.genB2A.eval(), self.disGA.eval(), self.disGB.eval(), self.disLA.eval(), self.disLB.eval()
                with torch.no_grad():
                    for _ in range(train_sample_num):
                        try:
                            real_A, _ = trainA_iter.next()
                        except:
                            trainA_iter = iter(self.trainA_loader)
                            real_A, _ = trainA_iter.next()

                        try:
                            real_B, _ = trainB_iter.next()
                        except:
                            trainB_iter = iter(self.trainB_loader)
                            real_B, _ = trainB_iter.next()
                        real_A, real_B = real_A.to(self.device), real_B.to(self.device)

                        fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                        fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                        fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                        fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                        fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                        fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

                        A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                                   cam(tensor2numpy(fake_A2A_heatmap[0]), (self.img_size, self.img_w)),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                                   cam(tensor2numpy(fake_A2B_heatmap[0]), (self.img_size, self.img_w)),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                                   cam(tensor2numpy(fake_A2B2A_heatmap[0]), (self.img_size, self.img_w)),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

                        B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                                   cam(tensor2numpy(fake_B2B_heatmap[0]), (self.img_size, self.img_w)),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                                   cam(tensor2numpy(fake_B2A_heatmap[0]), (self.img_size, self.img_w)),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                                   cam(tensor2numpy(fake_B2A2B_heatmap[0]), (self.img_size, self.img_w)),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

                    for _ in range(test_sample_num):
                        try:
                            real_A, _ = testA_iter.next()
                        except:
                            testA_iter = iter(self.testA_loader)
                            real_A, _ = testA_iter.next()

                        try:
                            real_B, _ = testB_iter.next()
                        except:
                            testB_iter = iter(self.testB_loader)
                            real_B, _ = testB_iter.next()
                        real_A, real_B = real_A.to(self.device), real_B.to(self.device)

                        fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                        fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                        fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                        fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                        fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                        fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

                        A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                                   cam(tensor2numpy(fake_A2A_heatmap[0]), (self.img_size, self.img_w)),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                                   cam(tensor2numpy(fake_A2B_heatmap[0]), (self.img_size, self.img_w)),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                                   cam(tensor2numpy(fake_A2B2A_heatmap[0]), (self.img_size, self.img_w)),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

                        B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                                   cam(tensor2numpy(fake_B2B_heatmap[0]), (self.img_size, self.img_w)),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                                   cam(tensor2numpy(fake_B2A_heatmap[0]), (self.img_size, self.img_w)),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                                   cam(tensor2numpy(fake_B2A2B_heatmap[0]), (self.img_size, self.img_w)),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'A2B_%07d.png' % step), A2B * 255.0)
                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'B2A_%07d.png' % step), B2A * 255.0)
                self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()
                torch.cuda.empty_cache()

            if step % self.save_freq == 0:
                self.save(os.path.join(self.result_dir, self.dataset, 'model'), step)

            if step % 1000 == 0:
                params = {}
                params['genA2B'] = self.genA2B.state_dict()
                params['genB2A'] = self.genB2A.state_dict()
                params['disGA'] = self.disGA.state_dict()
                params['disGB'] = self.disGB.state_dict()
                params['disLA'] = self.disLA.state_dict()
                params['disLB'] = self.disLB.state_dict()
                torch.save(params, os.path.join(self.result_dir, self.dataset + '_params_latest.pt'))

    def save(self, dir, step):
        params = {}
        params['genA2B'] = self.genA2B.state_dict()
        params['genB2A'] = self.genB2A.state_dict()
        params['disGA'] = self.disGA.state_dict()
        params['disGB'] = self.disGB.state_dict()
        params['disLA'] = self.disLA.state_dict()
        params['disLB'] = self.disLB.state_dict()
        torch.save(params, os.path.join(dir, self.dataset + '_params_%07d.pt' % step))

    def load(self, dir, step):
        params = torch.load(os.path.join(dir, self.dataset + '_params_%07d.pt' % step))
        self.genA2B.load_state_dict(params['genA2B'])
        self.genB2A.load_state_dict(params['genB2A'])
        self.disGA.load_state_dict(params['disGA'])
        self.disGB.load_state_dict(params['disGB'])
        self.disLA.load_state_dict(params['disLA'])
        self.disLB.load_state_dict(params['disLB'])

    def test(self):
        if self.resume_iter > 0:
            checkpoint_path = os.path.join(self.result_dir, self.dataset, 'model',
                                           self.dataset + '_params_%07d.pt' % self.resume_iter)
            if os.path.exists(checkpoint_path):
                self.load(os.path.join(self.result_dir, self.dataset, 'model'), self.resume_iter)
                print(" [*] Load SUCCESS")
                iter = self.resume_iter
            else:
                print(" [*] Load FAILURE")
                return
        else:
            model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pt'))
            if not len(model_list) == 0:
                model_list.sort()
                iter = int(model_list[-1].split('_')[-1].split('.')[0])
                self.load(os.path.join(self.result_dir, self.dataset, 'model'), iter)
                print(" [*] Load SUCCESS")
            else:
                print(" [*] Load FAILURE")
                return

        self.genA2B.eval(), self.genB2A.eval()
        with torch.no_grad():
            for n, (real_A, _) in enumerate(self.testA_loader):
                real_A = real_A.to(self.device)

                with autocast(enabled=self.amp):
                    fake_A2B, _, _ = self._call(self.genA2B, real_A)

                out_A2B = RGB2BGR(tensor2numpy(denorm(fake_A2B[0])))

                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', 'A2B_%d.png' % (n + 1)), out_A2B * 255.0)

            for n, (real_B, _) in enumerate(self.testB_loader):
                real_B = real_B.to(self.device)

                with autocast(enabled=self.amp):
                    fake_B2A, _, _ = self._call(self.genB2A, real_B)

                out_B2A = RGB2BGR(tensor2numpy(denorm(fake_B2A[0])))

                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', 'B2A_%d.png' % (n + 1)), out_B2A * 255.0)
        torch.cuda.empty_cache()
