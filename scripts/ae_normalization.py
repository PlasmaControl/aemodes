from tqdm.auto import tqdm
from aemodes.utils.dataset import load_dataset

data_path = '/scratch/gpfs/nc1514/aemodes/data/co2_250_detector.pkl'
# mean, std = 0.0523, 0.0654

train_dataset, valid_dataset = load_dataset(data_path)

mean, std = 0, 0
for i in tqdm(range(len(train_dataset))):
    inp = train_dataset[i]['X']
    mean += inp.mean()
    std += inp.std()
mean /= len(train_dataset)
std /= len(train_dataset)
print(mean, std)