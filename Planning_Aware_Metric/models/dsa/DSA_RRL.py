import torch
import torch.nn as nn

class custom_loss(nn.Module):
    def __init__(self, time, weights, supervised=False):
        super(custom_loss, self).__init__()
        if isinstance(weights,list):
            weights = torch.tensor(weights)

        self.cel =  nn.CrossEntropyLoss(weight = weights,reduction = 'none')
        self.time = time
        self.supervised = supervised
        if supervised:
            self.register_buffer('supervised_weight',torch.tensor([1.0]),persistent=False)
            self.obj_cel =  nn.BCELoss()
            self.m = nn.Sigmoid()

    def forward(self, outputs, targets, obj_pred=None, obj_targets=None):
        # targets: b (True or false)
        # outputs: txbx2
        # targets = targets.long() # convert to 0 or 1
        # outputs = outputs.permute(1,0,2) #bxtx2
        device = outputs.device
        outputs = outputs.permute(1,0,2)
        loss = torch.tensor(0.0).to(device)
        obj_loss = torch.tensor(0.0).to(device)
        for i,pred in enumerate(outputs):
            #bx2
            temp_loss = self.cel(pred,targets) # b
            exp_loss = torch.multiply(torch.exp(torch.tensor(-max(0,self.time-i-1) / 15.0)), temp_loss)
            exp_loss = torch.multiply(exp_loss,targets[:,1])
            loss = loss + torch.mean(temp_loss+exp_loss)

            if self.supervised:
                tmp = self.obj_cel(self.m(obj_pred[:,i]),obj_targets[:,i])
                tmp = self.supervised_weight * tmp
                # print("Loss:",obj_loss.mean(), temp_loss.mean())
                # print("obj_loss:",obj_loss,"collision_loss:",loss)
                obj_loss = obj_loss + tmp
        return {'collision_loss':loss,'obj_loss': obj_loss,'total_loss':loss+obj_loss}

class Baseline_SA(nn.Module):
    def __init__(self,backbone,n_frame, object_num, img_features_size=256, hidden_layer_size=256, lstm_size=512,supervised=False,zeros_object=False,intention=False,state=False):
        super(Baseline_SA, self).__init__()
        self.backbone = backbone
        self.n_frame = n_frame
        self.object_num = object_num
        self.supervised = supervised
        self.state = state
        if intention:
            self.intention = nn.Sequential(
                nn.Linear(9,hidden_layer_size//4),
                nn.ReLU(inplace=True), 
                nn.Linear(hidden_layer_size//4,hidden_layer_size//2),
                nn.ReLU(inplace=True), 
                )
        if supervised:
            self.supervised_obj = nn.Sequential(
            nn.Linear(hidden_layer_size,hidden_layer_size//2,bias=True),
            nn.ReLU(inplace=True), 
            nn.Linear(hidden_layer_size//2,1,bias=False),
        )
        if state:
            self.state_features = nn.Sequential(
                nn.Linear(2,hidden_layer_size//4),
                nn.ReLU(inplace=True), 
                nn.Linear(hidden_layer_size//4,hidden_layer_size//2),
                nn.ReLU(inplace=True), 
                )

        self.features_size = img_features_size
        self.zeros_object = zeros_object
        self.lstm_layer_size = lstm_size

        self.object_layer = nn.Linear(img_features_size, hidden_layer_size//2) if state  else nn.Linear(img_features_size, hidden_layer_size)
        self.frame_layer = nn.Linear(img_features_size//2, hidden_layer_size//2) if intention  else nn.Linear(img_features_size, hidden_layer_size)
        self.drop = nn.Dropout(p=0.3)
        self.lstm = nn.LSTMCell(hidden_layer_size*2, lstm_size)
        self.output_layer = nn.Linear(lstm_size, 2)
        
        self.object_attention = nn.Sequential(
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(inplace=True), 
        )
        self.h_prev = nn.Sequential(
            nn.Linear(lstm_size, hidden_layer_size,bias=False),
            nn.ReLU(inplace=True), 
        )
        self.to_attention = nn.Sequential(
            nn.Linear(hidden_layer_size,hidden_layer_size//2,bias=True),
            nn.ReLU(inplace=True), 
            nn.Linear(hidden_layer_size//2,1,bias=False),
        )

    def step(self, fusion, hx, cx):

        hx, cx = self.lstm(self.drop(fusion), (hx, cx))

        return self.output_layer(hx), hx, cx,#self.risky_object(hx)

    def forward(self, img, bbox,intention=None,state=None):
        """
            img: b t 3 H W
        """
        device = img.device
        batch_size, frames = img.shape[:2]
        hx = torch.zeros((batch_size, self.lstm_layer_size)).to(device)
        cx = torch.zeros((batch_size, self.lstm_layer_size)).to(device)
        out = []
        all_alphas = []
        all_obj = [] if self.supervised else None
        frame_features, obj_features = self.backbone(img,bbox)
        if intention is not None:
            intention = self.intention(intention)
        if self.zeros_object:
            zeros_object =  torch.sum(obj_features.permute(1,2,0,3),3).eq(0) # t x n x b
            zeros_object = ~zeros_object 
            zeros_object = zeros_object.float()
        
        for i in range(frames):
            img = frame_features[:,i]
            img = self.frame_layer(img)
            object = obj_features[:,i].permute(1,0,2) # n b c
            object = self.object_layer(object) # n b h
            if self.state:
                state_ = self.state_features(state[:,i].permute(1,0,2))
                object = torch.cat((object,state_),-1)
            if self.zeros_object:
                object = object*torch.unsqueeze(zeros_object[i],2)
            
            # object attention
            _object = self.object_attention(object)
            prev_object = self.h_prev(hx) + _object
            alphas = self.to_attention(prev_object).softmax(dim=0) # n b 1
            if self.supervised:
                obj_pred = self.supervised_obj(prev_object) # n b 1
                all_obj.append(obj_pred[:,:,0])
            if self.zeros_object:
                alphas = alphas * zeros_object[i]
            attention_list = alphas * object # n b h
            attention = attention_list.sum(0) # b h
            # state
            # concat frame & object
            if intention is not None:
                fusion = torch.cat((img,attention,intention),1)
            else:
                fusion = torch.cat((img,attention),1)
            pred,hx,cx = self.step(fusion,hx,cx)
            out.append(pred)
            all_alphas.append(alphas[:,:,0])

        out = torch.stack(out).permute(1,0,2)
        all_alphas = torch.stack(all_alphas).permute(2,0,1)
        if self.supervised:
            all_obj = torch.stack(all_obj).permute(2,0,1)
        # print(bbox[0,int(frames*0.8)])
        # print(all_alphas[0,int(frames*0.8)])
        return out, all_alphas, all_obj