# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 11:34:16 2022
"""
from pathlib import Path
from typing import List, Union, Tuple, Dict, Callable, Optional

import numpy as np
import pytorch_lightning as ptl
import SimpleITK as sitk
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from helper_functions import to_one_hot

def _normalize_array(x: np.ndarray) -> np.ndarray:
    """
    Brings the input array values into a range of [0, 1]

    Alternative options:
        - rescale to [-1, 1]
        - Instead of min/max take percentiles
        - z-score normalization
    """
    # There are other (and maybe better) ways of doing this, but this is the simplest
    min_val = np.min(x)
    max_val = np.max(x)

    return (x - min_val) / (max_val - min_val)

class WMHDataset(Dataset):
    def __init__(
        self, samples: List[Tuple[Path, Path, Path, int]], num_classes: int = 3
    ):
        """
        A WMHDataset is a custom PyTorch dataset built for use with the WMH
        challenge.

        :param samples: The list of samples for this dataset. This list
                        should contain, in order, a path to the T1, a
                        path to the T2, a path to the label image,
                        and a slice number.
        :param num_classes: How many classes does this dataset have?
        """
        super().__init__()
        self.samples: List[Tuple[Path, Path, Path, int]] = samples
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note that we have a list of tuples,
        # so every entry will return a
        # (Path, Path, Path, int).
        # If we fetch it like this, the first Path
        # will end up in the t1, the second in
        # the t2 and so on.
        t1, t2, gt, s = self.samples[index]

        reader = sitk.ImageFileReader()
        # The latest SimpleITK (v2.2.0) can deal with Path objects just fine.
        # However, all versions before that only accept string objects as path
        # input. Just to make sure this colab works for as many people as possible
        # we convert our path to a string here.
        reader.SetFileName(str(t1))
        reader.ReadImageInformation()

        image_size = reader.GetSize()

        if image_size[2] < s:                                       
            raise ValueError(
                f"Slice index {s} is greater than the maximum "
                f"z in image size {image_size}"
            )

        # Setting the extract index and size in a SimpleITK ImageFileReader
        # means that we don't read the whole image, only to throw away all
        # the slices that we don't need.
        # This way we will only read the slice that we want, and nothing else.

        # Where does our slice start?
        # Since we want the whole slice, at (0, 0, s)
        extract_index = (0, 0, s)

        # How much of the image do we want?
        # For this example, the whole slice, so image_size
        # But, we only want the single slice, so we set the last index 'z'
        # to 1.
        extract_size = list(image_size)
        extract_size[-1] = 1

        # Now we feed that to the reader
        reader.SetExtractIndex(extract_index)
        reader.SetExtractSize(extract_size)

        # And read all files
        t1 = reader.Execute()
        reader.SetFileName(str(t2))
        t2 = reader.Execute()
        reader.SetFileName(str(gt))
        gt = reader.Execute()

        # Unfortunately, we can't directly go from a sitk.Image to a
        # torch.Tensor, so we'll have to hop by a numpy array first.
        # We also squeeze the array to get rid of any dimensions of size 1
        t1 = sitk.GetArrayFromImage(t1).squeeze()
        t2 = sitk.GetArrayFromImage(t2).squeeze()
        gt = sitk.GetArrayFromImage(gt).squeeze()

        # Normalize the values of the input image to [0, 1]
        # Models tend to work better with input in this range.
        t1 = _normalize_array(t1)
        t2 = _normalize_array(t2)

        #remove NaN
        t1 = np.nan_to_num(t1,0)
        t2 = np.nan_to_num(t2,0)
        gt = np.nan_to_num(gt,0)
         
        # We use each contrast as a separate channel in our input image
        image = np.stack([t1, t2], axis=0)

        image = torch.from_numpy(image).type(torch.FloatTensor)
        gt = torch.from_numpy(gt).type(torch.FloatTensor)

        return image, to_one_hot(gt, self.num_classes)

class WMHPrefetchDataset(Dataset):
    def __init__(
        self, samples: List[Tuple[Path, Path, Path, int]], num_classes: int = 3
    ):
        """
        A WMHPrefetchDataset is a custom PyTorch dataset built for use with the WMH
        challenge.
        This version will load all the data into memory on initialization.

        :param samples: The list of samples for this dataset. This list
                        should contain, in order, a path to the T1, a
                        path to the T2, a path to the label image,
                        and a slice number.
        :param num_classes: How many classes does this dataset have?
        """
        super().__init__()
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.num_classes = num_classes

        for sample in tqdm(samples, "Prefetching dataset"):
            t1, t2, gt, s = sample
            
            reader = sitk.ImageFileReader()
            reader.SetFileName(str(t1))
            reader.ReadImageInformation()

            image_size = reader.GetSize()
            if image_size[2] < s:
                raise ValueError(
                    f"Slice index {s} is greater than the maximum "
                    f"z in image size {image_size}"
                )
            
            extract_index = (0, 0, s)
            extract_size = list(image_size)
            extract_size[-1] = 1
            reader.SetExtractIndex(extract_index)
            reader.SetExtractSize(extract_size)

            t1 = reader.Execute()
            reader.SetFileName(str(t2))
            t2 = reader.Execute()
            reader.SetFileName(str(gt))
            gt = reader.Execute()

            t1 = sitk.GetArrayFromImage(t1).squeeze()
            t2 = sitk.GetArrayFromImage(t2).squeeze()
            gt = sitk.GetArrayFromImage(gt).squeeze()

            t1 = _normalize_array(t1)
            t2 = _normalize_array(t2)
            
            #remove NaN
            t1 = np.nan_to_num(t1,0)
            t2 = np.nan_to_num(t2,0)
            gt = np.nan_to_num(gt,0)

            image = np.stack([t1, t2], axis=0)

            image = torch.from_numpy(image).type(torch.FloatTensor)
            gt = torch.from_numpy(gt).type(torch.FloatTensor)

            self.samples.append((image, to_one_hot(gt, self.num_classes)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[index]
    
    
class WMHTrainDataModule(ptl.LightningDataModule):
    def __init__(
        self,
        train_dataset_directory: Path,
        selected_train_subset: str = None,
        val_split: float = 0.1,
        batch_size: int = 8,
        num_workers: int = 4,
        num_classes: int = 3,
        use_prefetch: bool = False
    ):
        super(WMHTrainDataModule, self).__init__()

        self.train_dataset_directory: Path = train_dataset_directory
        self.selected_train_subset: str = selected_train_subset
        self.val_split: float = val_split

        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.num_classes: int = num_classes

        self.train_samples: List[Tuple[Path, Path, Path, int]] = []
        self.val_samples: List[Tuple[Path, Path, Path, int]] = []

        self.use_prefetch = use_prefetch

    def setup(self, stage: Optional[str] = None) -> None:
        found_train_samples: List[Tuple[Path, Path, Path, int]] = []
        found_val_samples: List[Tuple[Path, Path, Path, int]] = []

        #retreive the found samples for training
        if self.selected_train_subset is None:
            directories_to_search = [
                x for x in self.train_dataset_directory.glob("*") if x.is_dir()
            ]
        else:
            directories_to_search = [self.train_dataset_directory / self.selected_train_subset / "Train"]

        for directory in directories_to_search:
            patients = [x for x in directory.glob("*") if x.is_dir()]
            
            train_patients = []
            val_patients = []
            
            train_patients, val_patients = train_test_split(patients, test_size = self.val_split)
            
            for patient in train_patients:
                # These paths are known beforehand
                # Note that this notation, i.e.: path / "to" / "file.ext"
                # is only possible if you use Python's pathlib.Path
                label = patient / "PAD_wmh.nii.gz"
                t1 = patient / "PAD_T1.nii.gz"
                t2 = patient / "PAD_FLAIR.nii.gz"

                # How many slices does this patient have?
                reader = sitk.ImageFileReader()
                reader.SetFileName(str(t1))
                reader.ReadImageInformation()
                slices = reader.GetSize()[-1]

                for s in range(slices):
                    found_train_samples.append((t1, t2, label, s))
            
            for patient in val_patients:
                # These paths are known beforehand
                # Note that this notation, i.e.: path / "to" / "file.ext"
                # is only possible if you use Python's pathlib.Path
                label = patient / "PAD_wmh.nii.gz"
                t1 = patient / "PAD_T1.nii.gz"
                t2 = patient / "PAD_FLAIR.nii.gz"

                # How many slices does this patient have?
                reader = sitk.ImageFileReader()
                reader.SetFileName(str(t1))
                reader.ReadImageInformation()
                slices = reader.GetSize()[-1]

                for s in range(slices):
                    found_val_samples.append((t1, t2, label, s))

        #split the validation and test samples
        self.train_samples = found_train_samples 
        self.val_samples = found_val_samples
        

    def train_dataloader(self) -> DataLoader:
        if self.use_prefetch:
            dataset = WMHPrefetchDataset(self.train_samples, self.num_classes)
        else:
            dataset = WMHDataset(self.train_samples, self.num_classes)

        return DataLoader(
            dataset=dataset,
            shuffle=True,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

    def val_dataloader(self) -> DataLoader:
        if self.use_prefetch:
            dataset = WMHPrefetchDataset(self.val_samples, self.num_classes)
        else:
            dataset = WMHDataset(self.val_samples, self.num_classes)
        
        return DataLoader(
            dataset=dataset,
            shuffle=False,
            num_workers=self.num_workers,
            batch_size=self.batch_size // 4,
        )


class WMHTestDataModule(ptl.LightningDataModule):
    def __init__(
        self,
        test_dataset_directory: Path,
        selected_test_subset: str = None,
        batch_size: int = 8,
        num_workers: int = 4,
        num_classes: int = 3,
        use_prefetch: bool = False
    ):
        super(WMHTestDataModule, self).__init__()

        self.test_dataset_directory: Path = test_dataset_directory
        self.selected_test_subset: str = selected_test_subset

        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.num_classes: int = num_classes

        self.train_samples: List[Tuple[Path, Path, Path, int]] = []
        self.val_samples: List[Tuple[Path, Path, Path, int]] = []
        self.test_samples: List[Tuple[Path, Path, Path, int]] = []

        self.use_prefetch = use_prefetch

    def setup(self, stage: Optional[str] = None) -> None:
        found_test_samples: List[Tuple[Path, Path, Path, int]] = []
        
        #retreive the found samples for testing
        if self.selected_test_subset is None:
            directories_to_search = [
                x for x in self.test_dataset_directory.glob("*") if x.is_dir()
            ]
        else:
            directories_to_search = [self.test_dataset_directory / self.selected_test_subset / "Test"]

        for directory in directories_to_search:
            patients = [x for x in directory.glob("*") if x.is_dir()]
            for patient in patients:
                # These paths are known beforehand
                # Note that this notation, i.e.: path / "to" / "file.ext"
                # is only possible if you use Python's pathlib.Path
                label = patient / "PAD_wmh.nii.gz"
                t1 = patient / "PAD_T1.nii.gz"
                t2 = patient / "PAD_FLAIR.nii.gz"

                # How many slices does this patient have?
                reader = sitk.ImageFileReader()
                reader.SetFileName(str(t1))
                reader.ReadImageInformation()
                slices = reader.GetSize()[-1]

                for s in range(slices):
                    found_test_samples.append((t1, t2, label, s))
        
        self.test_samples = found_test_samples

    def test_dataloader(self) -> DataLoader:
        if self.use_prefetch:
            dataset = WMHPrefetchDataset(self.test_samples, self.num_classes)
        else:
            dataset = WMHDataset(self.test_samples, self.num_classes)

        return DataLoader(
            dataset=dataset,
            shuffle=False,
            num_workers=self.num_workers,
            batch_size=self.batch_size // 4,
        )
    

"""class WMHTrainDataModule(ptl.LightningDataModule):
    def __init__(
        self,
        train_dataset_directory: Path,
        selected_train_subset: str = None,
        val_split: float = 0.1,
        batch_size: int = 8,
        num_workers: int = 4,
        num_classes: int = 3,
        use_prefetch: bool = False
    ):
        super(WMHTrainDataModule, self).__init__()

        self.train_dataset_directory: Path = train_dataset_directory
        self.selected_train_subset: str = selected_train_subset
        self.val_split: float = val_split

        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.num_classes: int = num_classes

        self.train_samples: List[Tuple[Path, Path, Path, int]] = []
        self.val_samples: List[Tuple[Path, Path, Path, int]] = []

        self.use_prefetch = use_prefetch

    def setup(self, stage: Optional[str] = None) -> None:
        found_train_samples: List[Tuple[Path, Path, Path, int]] = []

        #retreive the found samples for training
        if self.selected_train_subset is None:
            directories_to_search = [
                x for x in self.train_dataset_directory.glob("*") if x.is_dir()
            ]
        else:
            directories_to_search = [self.train_dataset_directory / self.selected_train_subset]

        for directory in directories_to_search:
            patients = [x for x in directory.glob("*") if x.is_dir()]
            
            for patient in patients:
                # These paths are known beforehand
                # Note that this notation, i.e.: path / "to" / "file.ext"
                # is only possible if you use Python's pathlib.Path
                label = patient / "wmh.nii.gz"
                t1 = patient / "pre" / "T1.nii.gz"
                t2 = patient / "pre" / "FLAIR.nii.gz"

                # How many slices does this patient have?
                reader = sitk.ImageFileReader()
                reader.SetFileName(str(t1))
                reader.ReadImageInformation()
                slices = reader.GetSize()[-1]

                for s in range(slices):
                    found_train_samples.append((t1, t2, label, s))

        #split the validation and test samples
        self.train_samples, self.val_samples = train_test_split(
            found_train_samples, test_size=self.val_split
        )"""