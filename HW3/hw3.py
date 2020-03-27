import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import skimage.io as io
from skimage import data
from skimage.transform import resize
import os
from tqdm import tqdm

""" Root_dir = TRAIN_DIRECTORY_PATH = "/cifar10_train" """

class CifarDataset(torch.utils.data.Dataset):
  def __init__(self, root_dir,word_list):
    """Initializes a dataset containing images and labels."""
    super().__init__()
    self.data = np.array([])
    self.root_dir = root_dir
    
    for label,word in enumerate(word_list):     
      """
      Originally i thought loading all the images into one array would be a good idea
      but now I realize it's too memory inefficient.

      current_root = root_dir + word + "/*.png"
      col = io.imread_collection(current_root)
      n = len(np_col)
      np_col = io.concatenate_image(col).T
      label_col = np.array([i for _ in range(n)]).T
      """
      curr_data = os.listdir(os.path.join(self.root_dir,word))
      curr_data = np.array([[os.path.join(self.root_dir,word,x),label] for x in curr_data])
      if len(self.data) == 0:
        self.data = curr_data
      else:
        self.data = np.vstack((self.data,curr_data))
  
  def __len__(self):
    """Returns the size of the dataset."""
    return len(self.data)

  def __getitem__(self, index):
    """Returns the index-th data item of the dataset"""
    if index < 0 or index >= len(self.data):
      return None
    img_path , label = self.data[index][0] , self.data[index][1]
    """    
    Part 2
    image = io.imread(img_path)
    """
    
    """
    Part 3
    """
    image = resize(io.imread(img_path), (224,224)) # not resized in part 2
    return image, int(label)
# Model class
class MultiLayerPerceptron(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super().__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.Tanh()
    # self.drop = nn.Dropout(p = 0.2)
    self.fc2 = nn.Linear(hidden_size, num_classes)

  def forward(self, x):
    out = self.fc1(x.reshape(-1,3072))
    out = self.relu(out)
    # out = self.drop(out)
    out = self.fc2(out)
    return out
"""
# Part 3 training
def training(model,training_set, optimizer, criterion,device,num_epochs):
  model.train()

  # lambda2 = lambda epoch : 0.95**epoch
  # scheduler =  torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda = [lambda2]) 
  total_epochs = tqdm((range(num_epochs)))
  for epoch in total_epochs:
    for i, (images,labels) in enumerate(training_set):
      images = images.to(device)
      labels = labels.to(device)

      # print(type(images))
      outputs = model(images.float())
      loss = criterion(outputs, labels)

      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      if ((i+1) % 10) == 0:
        total_epochs.set_description('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                epoch + 1, num_epochs, i + 1, len(training_set), loss.item()))

# Part 2 Evaluate
def evaluate(model, test_set, device):
  model.eval()
  with torch.no_grad():
    correct = 0
    total = 0
    for images,labels in test_set:
      images = images.to(device)
      labels = labels.to(device)
      outputs = model(images.float())
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
  return (100 * correct / total)
"""

# Part 3 training
def training(model,training_set, optimizer, criterion,device,num_epochs):
  model.train()

  # lambda2 = lambda epoch : 0.95**epoch
  # scheduler =  torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda = [lambda2]) 
  total_epochs = tqdm((range(num_epochs)))
  for epoch in total_epochs:
    for i, (images,labels) in enumerate(training_set):
      images = images.to(device)
      labels = labels.to(device)
      images = images.permute(0,3,1,2)

      # print(type(images))
      outputs = model(images.float())
      loss = criterion(outputs, labels)

      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      if ((i+1) % 10) == 0:
        total_epochs.set_description('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                epoch + 1, num_epochs, i + 1, len(training_set), loss.item()))

#Part 3 Evaluate
def evaluate(model, test_set, device):
  model.eval()
  with torch.no_grad():
    correct = 0
    total = 0
    for images,labels in test_set:
      images = images.to(device)
      images = images.permute(0,3,1,2)
      labels = labels.to(device)
      outputs = model(images.float())
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
  return (100 * correct / total)

#constants
TRAIN_DIRECTORY_PATH = "cifar10_train"
TEST_DIRECTORY_PATH = "cifar10_test"
KEYWORD_LIST = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
INPUT_SIZE = 3072
NUM_CLASSES = 10

#hyperparameters
BATCH_SIZE = 100
NUM_EPOCHS = 3
lr = 0.000055
DEVICE = 'cuda:0'
HIDDEN_SIZE = 500

train_dataset = CifarDataset(TRAIN_DIRECTORY_PATH, KEYWORD_LIST)
test_dataset = CifarDataset(TEST_DIRECTORY_PATH, KEYWORD_LIST)

# Data Loader
train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset,
                                               batch_size = BATCH_SIZE,
                                               shuffle=True)

test_dataloader = torch.utils.data.DataLoader(dataset = test_dataset,
                                              batch_size = BATCH_SIZE,
                                              shuffle = False)
"""

#Part 2 models
model = MultiLayerPerceptron(INPUT_SIZE, HIDDEN_SIZE,NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adamax(model.parameters(), lr = lr, weight_decay = 0.55)

training(model, train_dataloader,optimizer,criterion,DEVICE,NUM_EPOCHS)
print("The model's accuracy was: {}".format(evaluate(model,test_dataloader,DEVICE)))
"""

# Part 3 
def set_parameter_requires_grad(model, FEATURE_EXTRACT):
  if FEATURE_EXTRACT:
    for param in model.parameters():
      param.requires_grad = False

FEATURE_EXTRACT = True # Freeze when Extract = True, finetune when Extract = False
pre_model = torch.hub.load('pytorch/vision:v0.5.0','mobilenet_v2',pretrained = True)
set_parameter_requires_grad(pre_model,FEATURE_EXTRACT)
pre_model.classifier[1] = torch.nn.Linear(in_features = pre_model.classifier[1].in_features, out_features = NUM_CLASSES) #fine tune last line
pre_model = pre_model.to(DEVICE)
pre_criterion = nn.CrossEntropyLoss()

params_to_update = pre_model.parameters()
if FEATURE_EXTRACT:
  params_to_update = []
  for name,param in pre_model.named_parameters():
    if param.requires_grad:
      params_to_update.append(param)

optimizer_pt3 = optim.SGD(params_to_update, lr, momentum = 0.9)
training(pre_model, train_dataloader,optimizer_pt3,pre_criterion,DEVICE,NUM_EPOCHS)
print("The Pre-trained model's accuracy was: {}".format(evaluate(pre_model,test_dataloader,DEVICE)))
