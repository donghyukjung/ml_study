from glob import glob
from data_preprocessing import *
from mnist_model import *
import torchvision.utils 
import matplotlib.pyplot as plt
DATA_PATH_LIST = '../../datasets/MNIST/mnist_png'
# LABEL_LIST=get_label_from_path(DATA_PATH_LIST)

def matplotlib_imshow(img, one_channel=False):
    fig,ax = plt.subplots(figsize=(16,8))
    ax.imshow(img.permute(1,2,0).numpy())
    plt.show()


if __name__ == "__main__": 
    learning_rate = 0.001
    training_epochs = 15
    batch_size = 100

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(777)

    if device == 'cuda':
        torch.cuda.manual_seed_all(777)

    model = CNN().to(device)

    transform = transforms.Compose( [ transforms.ToTensor(), ] ) 

    # train_data = mnistDataset(path=DATA_PATH_LIST, train=True, transform=transform) 
    # train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=False) 

    # test_data = mnistDataset(path=DATA_PATH_LIST, train=False, transform=transform) 
    # test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, drop_last=False) 
    mnist_train = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                          train=True, # True를 지정하면 훈련 데이터로 다운로드
                          transform=transforms.ToTensor(), # 텐서로 변환
                          download=True)

    mnist_test = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                         train=False, # False를 지정하면 테스트 데이터로 다운로드
                         transform=transforms.ToTensor(), # 텐서로 변환
                         download=True)
    data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

    total_batch=len(mnist_train)
    for epoch in range(training_epochs):
        avg_cost=0

        for X,Y in data_loader:
            X=X.to(device)
            Y=Y.to(device)

            optimizer.zero_grad()
            hypothesis = model(X)
            cost = criterion(hypothesis, Y)
            cost.backward()
            optimizer.step()

            avg_cost += cost / total_batch

        print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))


    with torch.no_grad():
        X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
        Y_test = mnist_test.test_labels.to(device)

        prediction = model(X_test)
        correct_prediction = torch.argmax(prediction, 1) == Y_test
        accuracy = correct_prediction.float().mean()
        print('Accuracy:', accuracy.item())
    # Visualize images
    # images= next(iter(train_dataloader))[0][:16]
    # img_grid = torchvision.utils.make_grid(images, nrow=8, normalize=True)
    # matplotlib_imshow(img_grid)   