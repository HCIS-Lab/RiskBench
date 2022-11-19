import torch
import torch.nn as nn
__all__ = [
    'Baseline_Jinkyu',
]

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class Baseline_Jinkyu(nn.Module):
    def __init__(self, inputs, time_steps=90, pretrained= True ):
        super(Baseline_Jinkyu, self).__init__()

        self.hidden_size = 512
        self.D = 64
        self.L = 240
        self.H = 512
        if inputs in ['camera', 'sensor', 'both']:
            self.with_camera = 'sensor' not in inputs
            self.with_sensor = 'camera' not in inputs
        else:
            raise(RuntimeError(
                'Unknown inputs of {}, '
                'supported inputs consist of "camera", "sensor", "both"',format(inputs)))
        self.time_steps = time_steps
        self.pretrained = pretrained

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2 ),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        #InceptionResNetV2(num_classes=1001)

        self.project_w = nn.Linear(self.D, self.D)
        self.w = nn.Linear(self.H, self.D)
        self.w_attn = nn.Linear(self.D, 1)

        self.camera_features = nn.Sequential(
            nn.Conv2d(1536, 20, kernel_size=1),
            nn.ReLU(inplace=True),
            Flatten(),
        )

        if self.with_camera and self.with_sensor:
            raise(RuntimeError('Sensor Data is not Input'))
        elif self.with_camera:
            self.fusion_size = self.D*self.L
        elif self.with_sensor:
            raise(RuntimeError('Sensor Data is not Input'))
        else:
            raise(RuntimeError('Inputs of camera and sensor cannot be both empty'))

        self.drop = nn.Dropout(p=0.1)
        self.lstm = nn.LSTMCell(self.fusion_size, self.hidden_size)


        self.steer_regressor = nn.Sequential(
            nn.Linear(self.hidden_size, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 1),
        )

        self.vel_regressor= nn.Sequential(
            nn.Linear(self.hidden_size, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 1),
        )

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('BasicConv2d') != -1:
                pass
            elif classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.001)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.001)
                m.bias.data.fill_(0.001)


    def step(self, camera_input, hx, cx):

        hx, cx = self.lstm(self.drop(camera_input), (hx, cx))

        return self.steer_regressor(self.drop(hx)), self.vel_regressor(self.drop(hx)), hx, cx

    def project_feature(self, feature):
        feature_flat = feature.view(-1, self.D)
        feature_proj = self.project_w(feature_flat)
        feature_proj = feature_proj.view(-1, self.L, self.D)

        return feature_proj

    def attntion_layer(self, features, features_proj, h):
        ####################################
        # features: b,1,L,D
        # features_proj: b,1,L,D
        ####################################
        #print(self.w(h).shape)
        h_attn = torch.tanh(features_proj+torch.unsqueeze(self.w(h),1)) # b,L,D
        out_attn = self.w_attn(h_attn.view(-1, self.D)).view(-1, self.L) #b,L
        alpha = nn.functional.softmax(out_attn) #b,L
        alpha_logp = nn.functional.log_softmax(out_attn)
        context = (features*torch.unsqueeze(alpha,2)).view(-1,self.L*self.D)# b,D

        return context, alpha, alpha_logp

    def forward(self, camera_inputs, device):

        batch_size = camera_inputs.shape[0]
        t = camera_inputs.shape[1]
        c = camera_inputs.shape[2]
        w = camera_inputs.shape[3]
        h = camera_inputs.shape[4]

        # initialize LSTM
        hx = torch.zeros((batch_size, self.hidden_size)).to(device)
        cx = torch.zeros((batch_size, self.hidden_size)).to(device)
        logit_steer_stack = []
        logit_vel_stack = []


        camera_inputs = self.backbone(camera_inputs.view(-1,c,w,h)) # camera_inputs :(bs,t,c,w,h)
        camera_inputs = camera_inputs.permute(0,3,1,2).contiguous()
        camera_inputs = camera_inputs.view(-1,64,240)
        camera_inputs = camera_inputs.permute(0,2,1).contiguous() #(bxt, self.L, self.D)

        features_proj = self.project_feature(camera_inputs) #(bxt, L, D)


        for l in range(0, self.time_steps):
            features_curr =  camera_inputs.view(batch_size,t,self.L, self.D)[:,l]
            features_proj_curr = features_proj.view(batch_size,t,self.L, self.D)[:,l]
            context, alpha, alpha_logp = self.attntion_layer(features_curr,features_proj_curr,hx)
            steer, vel, hx, cx = self.step(context, hx, cx)
            logit_steer_stack.append(steer)
            logit_vel_stack.append(vel)

        logit_steer_stack = torch.stack(logit_steer_stack).view(-1)
        logit_vel_stack = torch.stack(logit_vel_stack).view(-1) #t x batch

        return logit_steer_stack, logit_vel_stack


