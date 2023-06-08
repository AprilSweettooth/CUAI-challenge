import pandas as pd
import numpy as np
import _pickle as pkl
import bz2
from PIL import Image

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import os


floy_data_manager = None

# only keep separate holdout set for now, let them split and annotate data themselves
class FloyBrainMriDataset(Dataset):
    '''Exemplaric class, how the FloyDataManager can be used. Feel free to use this Dataset in the challenge,
    or adapt it to fit your needs.'''
    
    def __init__(self, split='train', val_size=0.2, transforms=None, data_path=None,available_annotations=1.0):
        '''
        Dataset class constructor
        Arguments:
            split: 'train', 'val' or 'test' (=holdout set)
            val_size: fraction of data to use for validation (fraction of whole *training* set!)
            transforms: transforms to apply to the data

            data_path: path to the 'kaggle_m3' folder. Only needs to be passed, 
                if no FloyDataManager was initialized, and the first dataset is initialized.
            available_annotations: fraction of available annotations to use. Same as with data_path:
                Only needed to initialize the FloyDataManager (if not initialized yet)
        '''

        global floy_data_manager
        if floy_data_manager is None:

            # the FloyDataManager saves a reference to itself upon construction.
            FloyDataManager(data_path, available_annotations=available_annotations)

        if not val_size == 0.0:
            train, val = train_test_split(floy_data_manager.train_df, test_size=val_size, random_state=42)
        else:
            train = floy_data_manager.train_df

        if(split == 'train'):
            self.df = train
        elif split == 'val':
            if(val is None):
                raise ValueError('No validation set available. Please adjust val_size!')
            else:
                self.df = val
        else:
            self.split = 'holdout'
            self.df = floy_data_manager.test_df

        self.transforms = transforms
        self.split = split
        
    def annotate_sample(self, idx):
        '''
        Annotate a sample, using the floy data manager
        '''
        global floy_data_manager

        # get correct ID for floy_data_manager
        data_manager_idx = self.df.iloc[idx].name
        floy_data_manager.annotate_sample(data_manager_idx)

    def submit_prediction(self, idx, mask):
        '''
        Submit a prediction, to the floy data manager
        '''
        global floy_data_manager

        # throw an error, if this is not the test set
        if(self.split != 'test') and (self.split != 'holdout'):
            raise ValueError('This is not the test set!')

        # get correct ID for floy_data_manager
        data_manager_idx = self.df.iloc[idx].name
        floy_data_manager.submit_prediction(data_manager_idx, mask)
    
    def get_available_annotation_count(self):
        '''Returns the number of available annotations left'''
        return floy_data_manager.get_available_annotation_count()

    def extract_result_file(self, path_out, team_name):
        floy_data_manager.extract_result_file(path_out, team_name)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        global floy_data_manager

        data_manager_idx = self.df.iloc[idx].name
        image, mask = floy_data_manager.retrieve_sample(data_manager_idx, dataset_part=self.split)

        if not (self.transforms is None):
           
            if(mask is None):
                augmented = self.transforms(image=image)
                image = augmented['image']
            else:
                augmented = self.transforms(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']   
 

        return image, mask









##### Please do not change the code below this line, as it is vital to keep our competition fair :) #####
# control data and annotation access to floy data
class FloyDataManager():
    '''This class keeps track of the annotations you request, 
    and facilitates data access. For possible implementation, 
    see the Dataset class above.
    
    Note: The FloyDataManager does not distinguish between 
    training and validation set. This is done in the Dataset class,
    and forces you to distribute your annotations wisely 
    between training and validation set!'''

    def __init__(self, data_path, available_annotations=1.0):
        '''Initialize the FloyDataManager.
        Arguments:
            data_path: path to the 'kaggle_m3' folder. Only needs to be passed,
                if no FloyDataManager was initialized, and the first dataset is initialized.
            available_annotations: fraction of available annotations to use. Use wisely, 
                and distribute between training and validation sets as you need'''

        global floy_data_manager
        floy_data_manager = self

        if(data_path is None):
            raise Exception('No data path passed. Please provide the data path to the "kaggle_3m" folder to initialize the FloyAnnotator')

        # If the Dataset class is initialized for the first time, load all data and prepare Dataframe
        self.data_path = data_path
        full_mri_data = pd.read_csv(os.path.join(data_path, 'floy_data.csv'))
        if('Unnamed: 0' in full_mri_data.columns):
            full_mri_data = full_mri_data.drop(columns=['Unnamed: 0'])
            
        self.train_df = full_mri_data[full_mri_data.dataset_part == 'train/val'].set_index('image_path', drop=True)
        self.train_df['annotation_requested'] = False

        self.test_df = full_mri_data[full_mri_data.dataset_part == 'holdout'].set_index('image_path', drop=True)
        self.test_df['prediction_available'] = False
        self.available_annotations = available_annotations

        # Initialize Logging, to allow for later analysis
        self.annotation_log = pd.DataFrame([], columns=['image', 'data_step'])
        self.data_step_counter = 0
        self.holdout_prediction_log = {}

        print("Successfully initialized FloyDataManager. {} annotations can be retrieved.".format(self.get_available_annotation_count()))


    def get_available_annotation_count(self, current=True):
        '''
        Returns the number of annotations left, that can be queried in this iteration.
        If current = False, returns the number of total annotations that can be retrieved 
        (including those who were already annotated)
        '''
        annotations_gathered = len(self.train_df[self.train_df.annotation_requested == True])
        annotations_available = int(len(self.train_df)*self.available_annotations)
        if(current):
            return annotations_available - annotations_gathered
        else:
            return annotations_available

    
    def annotate_sample(self, image_path):
        '''
        Annotate a sample with the given index. 
        If no more annotations are available, raise an Exception
        '''
        if not image_path in self.train_df.index:
            raise Exception('Image path not in training set')

        # do nothing, if image is already annotated
        if self.train_df.loc[image_path, 'annotation_requested']:
            return

        if self.get_available_annotation_count() == 0:
            raise Exception('Illegal annotation requested for idx {}. \nAnnotation limit ({}) reached.'.format(image_path, 
                            int(len(self.train_df)*self.available_annotations)))
        else:
            self.train_df.loc[image_path,'annotation_requested'] = True
            
            # Attach the new annotation to the annotation log dataframe
            annotation_log_entry = [image_path, self.data_step_counter]
            self.annotation_log = self.annotation_log.append(pd.Series(annotation_log_entry, index=self.annotation_log.columns), ignore_index=True)

    def submit_prediction(self, image_path, prediction):
        '''
        Submit the prediction for the given sample.
        Arguments:
            idx: index of the sample
            prediction: 256x256 numpy array with the predicted mask
        '''
        # check if the prediction meets the requirements
        if not type(prediction) == np.ndarray:
            raise ValueError('Prediction must be a numpy array')
        if not prediction.shape == (256, 256):
            raise ValueError('Prediction must be a 256x256 numpy array')

        prediction[np.nonzero(prediction < 0.5)] = 0.0
        prediction[np.nonzero(prediction >= 0.5)] = 1.0

        self.test_df.loc[image_path, 'prediction_available'] = True
        self.holdout_prediction_log[image_path] = prediction

    def extract_result_file(self, path_out, team_name):
        # check if all test samples have been submitted
        if len(self.test_df[self.test_df.prediction_available == False]) > 0:
            raise Exception('Not all holdout samples have been submitted!\n Missing indices: {}'.format(
                self.test_df[self.test_df.prediction_available == False].index))

        pickle_out_obj = {}

        pickle_out_obj['holdout_predictions'] = self.holdout_prediction_log
        pickle_out_obj['annotation_log'] = self.annotation_log
        pickle_out_obj['test_df'] = self.test_df
        pickle_out_obj['available_annotations'] = self.available_annotations
        pickle_out_obj['team_name'] = team_name

        # save pickle file
        with bz2.BZ2File(path_out, 'wb') as f:
            pkl.dump(pickle_out_obj, f)

        

    def retrieve_sample(self, image_path, dataset_part='train'):
        '''
        Retrieve a sample with the given index.
        Define the part of the dataset, either holdout, train or val'''

        self.data_step_counter += 1

        if (dataset_part == 'holdout') or (dataset_part == 'test'):
            image = Image.open(os.path.join(self.data_path, self.test_df.loc[image_path,'patient'], image_path))
            image = np.array(image)

            if(os.path.exists(os.path.join(self.data_path, self.test_df.loc[image_path,'patient'], image_path[:-4]+'_mask.tif'))):
                mask = Image.open(os.path.join(self.data_path, self.test_df.loc[image_path,'patient'], image_path[:-4]+'_mask.tif'))
                mask = np.array(mask)
                return image, mask
            else:
                return image, None


        # if the data is not holdout, it is train or val
        image = Image.open(os.path.join(self.data_path, self.train_df.loc[image_path,'patient'], image_path))
        image = np.array(image)
        
        mask = Image.open(os.path.join(self.data_path, self.train_df.loc[image_path,'patient'], image_path[:-4]+'_mask.tif'))
        mask = np.array(mask)

        if not self.train_df.loc[image_path,'annotation_requested']:
            mask = None

        return image, mask