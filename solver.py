import os
from os.path import join as ospj
import time
import datetime
from munch import Munch
from torchvision.utils import save_image
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import build_model
from checkpoint import CheckpointIO
from dataset import InputFetcher
import utils


class Solver(nn.Module):
    def __init__(self,args,loader):
        super(Solver,self).__init__()
        self.args=args
        #self.logger=logger
        self.loaders=loader
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nets,self.nets_ema=build_model(args)
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)
        

        if args.mode == 'train':
            self.optims = Munch()
            for net in self.nets.keys():
                if net == 'fan':
                    continue
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.f_lr if net == 'mapping_network' else args.lr,
                    betas=[args.beta1, args.beta2],
                    weight_decay=args.weight_decay)

            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets.ckpt'), data_parallel=False, **self.nets),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), data_parallel=False, **self.nets_ema),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_optims.ckpt'), **self.optims)]
        else:
            self.ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets.ckpt'), data_parallel=False, **self.nets_ema)]

        self.to(self.device)
        for name, network in self.named_children():
            # Do not initialize the FAN parameters
            if ('ema' not in name):
                print('Initializing %s...' % name)
                network.apply(utils.he_init)
        self.bce_loss=nn.BCEWithLogitsLoss()
        self.l1_loss=nn.L1Loss()

    def savecheckpoint(self,step):
        for ckpt in self.ckptios:
            ckpt.save(step)
    
    def loadcheckpoint(self,step):
        for ckpt in self.ckptios:
            ckpt.load(step)
    
    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()
    
    def r1_reg(self,d_out, x_in):
    # zro-centered gradient penalty for real images
        batch_size = x_in.size(0)
        x_in.requires_grad_()
        grad_dout = torch.autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert(grad_dout2.size() == x_in.size())
        reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
        return reg
    
    def forward_G(self,x_src,s_target):
        self.x_fake=self.nets.generator(x_src,s_target)
    
    def forward_mapping_network(self,z_trg,z_trg2,y_trg):
        self.s_trg_map=self.nets.mapping_network(z_trg,y_trg)
        self.s_trg2_map=self.nets.mapping_network(z_trg2,y_trg)
    
    def encode_style(self,x_ref,x_ref2,y_trg):
        self.s_trg_enc=self.nets.style_encoder(x_ref,y_trg)
        self.s_trg2_enc=self.nets.style_encoder(x_ref2,y_trg)

    def Backward_D(self,x_src,y_org,y_target,z_target=None,x_ref=None):
        assert (z_target is None) != (x_ref is None)
        #print('forward_pass_real_discriminator')
        out=self.nets.discriminator(x_src,y_org)
        self.d_real_loss=self.bce_loss(out,torch.ones_like(out))
        #self.d_r1_reg=self.r1_reg(out,x_src)
        if z_target is not None:
            s_trg=self.nets.mapping_network(z_target,y_target)
            self.forward_G(x_src,s_trg)
        else:
            s_trg=self.nets.style_encoder(x_ref,y_target)
            self.forward_G(x_src,s_trg)
        
        #print('forward_pass_fake_discriminator')
        out=self.nets.discriminator(self.x_fake.detach(),y_target)
        self.d_fake_loss=self.bce_loss(out,torch.zeros_like(out))
        self.d_loss=self.d_real_loss+self.d_fake_loss
        
    def Backward_G(self,x_src,y_org,y_target,z_targets=None,x_refs=None):
        assert (z_targets is None) != (x_refs is None)
        if z_targets is not None:
            z_target,z_target2=z_targets
            s_trg=self.nets.mapping_network(z_target,y_target)
            s_trg2=self.nets.mapping_network(z_target2,y_target)
        if x_refs is not None:
            x_ref,x_ref2=x_refs
            s_trg=self.nets.style_encoder(x_ref,y_target)
            s_trg2=self.nets.style_encoder(x_ref2,y_target)
        #print('generative loss')
        self.forward_G(x_src,s_trg)
        out = self.nets.discriminator(self.x_fake, y_target)
        self.g_adv_loss = self.bce_loss(out,torch.ones_like(out))
        # style reconstruction loss
        #print('styleloss')
        s_pred = self.nets.style_encoder(self.x_fake, y_target)
        self.style_loss = self.l1_loss(s_trg,s_pred)
        # diversity sensitive loss
        #print('ds loss')
        x_fake2=self.nets.generator(x_src,s_trg2,masks=None).detach()
        self.ds_loss = self.l1_loss(x_fake2,self.x_fake)
        # cycle-consistency loss
        #print('cycle loss')
        s_org = self.nets.style_encoder(x_src,y_org)
        x_rec = self.nets.generator(self.x_fake, s_org, masks=None).detach()
        self.cycle_loss = self.l1_loss(x_src,x_rec)
        self.g_loss = self.g_adv_loss + self.args.lambda_sty * self.style_loss -self.args.lambda_ds * self.ds_loss + self.args.lambda_cyc * self.cycle_loss
        
    def moving_average(self,nets,nets_ema,beta):
        for param, param_test in zip(nets.parameters(), nets_ema.parameters()):
            param_test.data = torch.lerp(param.data, param_test.data, beta)
    def Train_network(self):
        print('into training')
        fetcher = InputFetcher(self.loaders.src,None, self.args.latent_dim, 'train')
        fetcher_val = InputFetcher(self.loaders.val, None, self.args.latent_dim, 'val')
        inputs_val = next(fetcher_val)
    # resume training if necessary
        if self.args.resume_iter > 0:
            self.loadcheckpoint(self.args.resume_iter)
        
    # remember the initial value of ds weight
        initial_lambda_ds = self.args.lambda_ds
        print("Starts Training")
        start_time=time.time()
        for i in range(self.args.resume_iter,self.args.total_iters):
            inputs=next(fetcher)
            x_src,y_org=inputs.x_src,inputs.y_src
            x_ref,x_ref2,y_target=inputs.x_ref,inputs.x_ref2,inputs.y_ref
            z_trg,z_trg2=inputs.z_trg,inputs.z_trg2
            #self.forward_mapping_network(z_trg,z_trg2,y_target)
            #self.encode_style(x_ref,x_ref2,y_target)
            loss=dict()
            self.Backward_D(x_src,y_org,y_target,z_target=z_trg)
            self._reset_grad()
            self.d_loss.backward()
            self.optims.discriminator.step()
            loss['D_loss_latent']=self.d_loss.item()
            self.Backward_D(x_src,y_org,y_target,None,x_ref=x_ref)
            self._reset_grad()
            self.d_loss.backward()
            self.optims.discriminator.step()
            loss['D_loss_reference']=self.d_loss.item()
            self.Backward_G(x_src,y_org,y_target,z_targets=[z_trg,z_trg2])
            self._reset_grad()
            self.g_loss.backward()
            self.optims.generator.step()
            self.optims.mapping_network.step()
            self.optims.style_encoder.step()
            loss['G_loss_latent']=self.g_loss.item()
            self.Backward_G(x_src,y_org,y_target,None,x_refs=[x_ref,x_ref2])
            self._reset_grad()
            self.g_loss.backward()
            self.optims.generator.step()
            loss['G_loss_reference']=self.g_loss.item()
            loss['style_loss']=self.style_loss.item()
            
            #self.moving_average(self.nets.generator, self.nets_ema.generator, beta=0.999)
            #self.moving_average(self.nets.mapping_network, self.nets_ema.mapping_network, beta=0.999)
            #self.moving_average(self.nets.style_encoder, self.nets_ema.style_encoder, beta=0.999)
            if (i+1) % self.args.print_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i+1, self.args.total_iters)
                print("{} Iteration [{}/{}] g_loss_latent: {:.6f}, d_loss_latent: {:.6f}, g_loss_ref: {:.6f}, d_loss_ref: {:.6f},elapse: {} seconds".
                                format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i, self.args.total_iters,
                                    loss['G_loss_latent'],loss['D_loss_latent'],loss['G_loss_reference'],loss['D_loss_reference'],elapsed))
            if (i+1) % self.args.sample_step == 0:
                os.makedirs(self.args.sample_dir, exist_ok=True)
                utils.debug_image(self.nets, self.args, inputs=inputs_val, step=i+1)
        # save model checkpoints
            if (i+1) % self.args.save_step == 0:
                self.savecheckpoint(step=i+1)
            
        
    
    @torch.no_grad()
    def sample(self):
        os.makedirs(self.args.result_dir, exist_ok=True)
        self.loadcheckpoint(self.args.resume_iter)
        self.angles=['000','018','036','054','072','090','108','126','144','162','180']
        self.states=['nm-01','nm-02','nm-03','nm-04','nm-05','nm-06','bg-01','bg-02','cl-01','cl-02']
        for _,(x_src_img,id,k,j,x_ref_img,y_ref) in enumerate(self.loaders.test):
            x_src=x_src_img.to(self.device)
            x_ref_img=x_ref_img.to(self.device)
            y_ref=y_ref.to(self.device)
            result=ospj(self.args.result_dir,self.angles[y_ref.item()])
            if not os.path.exists(result):
                os.makedirs(result)
            id=id[0]
            cond=self.states[k.item()]
            angle=self.angles[j.item()]
            result=ospj(self.args.result_dir,self.angles[y_ref.item()],id,cond,angle)
            if not os.path.exists(result):
                os.makedirs(result)
            fname=ospj(result,id+'-'+cond+'-'+angle+'.png')
            s_trg=self.nets.style_encoder(x_ref_img,y_ref)
            x_fake = self.nets.generator(x_src_img, s_trg, masks=None)
            img = utils.denorm(x_fake.data.cpu())
            save_image(img,fname,nrow=1)
            print("Image Saved ",id,cond,angle,self.angles[y_ref.item()])
    

        