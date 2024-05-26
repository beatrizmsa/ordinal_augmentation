from skimage.io import imread
import pandas as pd
import os

class Smear:
    # https://mde-lab.aegean.gr/index.php/downloads/
    allow_vflips = True
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


class Adience:
    # http://www.cslab.openu.ac.il/download/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/ 
    # user = adiencedb
    # passeword = adience
    allow_vflips = False
    num_classes = 8
    classes = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)',
               '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
    dname = 'aligned'

    def __init__(self, root, transform=None):
        self.root = os.path.join(root, 'Adience')
        
        process = TxtToCsvConverter(root = self.root, dname = self.dname)
        
        df = process.process_files()
        header_list = df.columns.tolist()
        self.files = [(row[header_list[2]], row[header_list[1]]) for _, row in df.iterrows() if row[header_list[2]] in self.classes]
        self.labels = [self.classes.index(klass) for klass, _ in self.files]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        _,fname = self.files[i]
        label = self.labels[i]
        image = imread(os.path.join(self.root,fname))
        if self.transform:
            image = self.transform(image)
        return image, label

class TxtToCsvConverter:
    def __init__(self, root, delimiter='\t', dname = None):
        self.root = root
        self.delimiter = delimiter
        self.dname = dname

    def process_files(self):
        dataframes = []

        file_list = [fname for fname in os.listdir(self.root) if fname.endswith('.txt')]
        
        for file in file_list:
            file_path = os.path.join(self.root, file)
            df = pd.read_csv(file_path, delimiter=self.delimiter)
            selected_columns = df[['user_id','original_image', 'age']].dropna(subset=['age'])
            dataframes.append(selected_columns)

        df = pd.concat(dataframes, ignore_index=True)

        df.drop_duplicates(subset=['original_image'], inplace=True)

        existing_files = []

        for index, row in df.iterrows():
            folder, fname = row['user_id'], row['original_image']
            folder_path = os.path.join(self.root, self.dname, folder)
            file_exists = False
            files = os.listdir(folder_path)
            for image in files:
                if image.endswith(fname):
                    df.at[index, 'original_image'] = f"{os.path.join(self.dname, folder, image)}"
                    file_exists = True
                    break
            if not file_exists:
                existing_files.append(index)

        df = df.drop(index=existing_files)

        return df

if __name__ == '__main__':
    ds = Adience('/nas-ctm01/homes/bmsa/data')
    x, y = ds[0]
    print('x:', type(x), x.shape, x.dtype, x.min(), x.max())
    print('y:', y)

