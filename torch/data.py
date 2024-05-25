from skimage.io import imread
import pandas as pd
import os

class Smear:
    # https://mde-lab.aegean.gr/index.php/downloads/
    num_classes = 7
    classes = ['normal_superficiel', 'normal_intermediate', 'normal_columnar', 'light_dysplastic',
               'moderate_dysplastic', 'severe_dysplastic', 'carcinoma_in_situ']

    def __init__(self, root, transform=None):
        self.root = os.path.join(root, 'smear2005', 'New database pictures')
        self.files = [(klass, fname) for klass in self.classes for fname in os.listdir(os.path.join(self.root, klass))
            if fname.endswith('.BMP')]
        self.labels = [self.classes.index(klass) for klass, _ in self.files]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        dname, fname = self.files[i]
        label = self.labels[i]
        image = imread(os.path.join(self.root, dname, fname))
        if self.transform:
            image = self.transform(image)
        return image, label

# if __name__ == '__main__':
#     ds = Smear('/nas-ctm01/homes/bmsa/data')
#     x, y = ds[0]
#     print('x:', type(x), x.shape, x.dtype, x.min(), x.max())
#     print('y:', y)


class Adience:
    # http://www.cslab.openu.ac.il/download/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/ 
    # user = adiencedb
    # passeword = adience
    num_classes = 8
    classes = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)',
               '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
    
    def __init__(self, root, transform=None):
        self.root = os.path.join(root, 'Adience')
        
        process = TxtToCsvConverter(self.root)
        process.process_files()
        
        df = pd.read_csv(os.path.join(self.root,'Adience.csv'))
        header_list = df.columns.tolist()
        self.files = [(row[header_list[2]], row[header_list[1]]) for _, row in df.iterrows() if row[header_list[2]] in self.classes]
        self.folder = [ (row[header_list[0]]) for _, row in df.iterrows()]
        self.labels = [self.classes.index(klass) for klass, _ in self.files]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        dname = 'aligned'
        folder = self.folder[i]
        _, fname = self.files[i]
        label = self.labels[i]
        image = imread(os.path.join(self.root, dname, folder, fname))
        if self.transform:
            image = self.transform(image)
        return image, label

class TxtToCsvConverter:
    def __init__(self, root, delimiter='\t'):
        self.root = root
        self.delimiter = delimiter
        self.output_filename = os.path.join(self.root,"Adience.csv")

    def process_files(self):
        dataframes = []

        file_list = [fname for fname in os.listdir(self.root) if fname.endswith('.txt')]
        
        for file in file_list:
            file_path = os.path.join(self.root, file)
            df = pd.read_csv(file_path, delimiter=self.delimiter)
            selected_columns = df[['user_id','original_image', 'age']].dropna(subset=['age'])
            dataframes.append(selected_columns)

        combined_df = pd.concat(dataframes, ignore_index=True)

        combined_df.drop_duplicates(subset=['original_image'], inplace=True)

        combined_df.to_csv(self.output_filename, index=False)

if __name__ == '__main__':
    ds = Adience('/nas-ctm01/homes/bmsa/data')
    x, y = ds[0]
    print('x:', type(x), x.shape, x.dtype, x.min(), x.max())
    print('y:', y)

