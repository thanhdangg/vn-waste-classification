class WasteDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        try:
            image = Image.open(img_path).convert('RGB')
        except UnidentifiedImageError:
            print(f"⚠️ Skipping corrupted image: {img_path}")
            return self.__getitem__((idx + 1) % len(self.samples))
        if self.transform:
            image = self.transform(image)
        return image, label