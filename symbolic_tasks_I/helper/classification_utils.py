"""
Utilities shared by the notebooks of the classification tasks
"""
import glob
import os
import shutil
import tempfile
import zipfile
from typing import List, Optional, Union

import numpy as np
import matplotlib.pyplot as plt


def _extract_zip(zip_fn: Union[str, os.PathLike], target_dir: Union[str, os.PathLike]) -> None:
    """Extract a zip file, ignoring the files that macOS adds to zip files (__MACOSX, ._*, .DS_Store)"""
    with zipfile.ZipFile(zip_fn) as zf:
        for member in zf.infolist():
            parts = member.filename.split("/")
            if "__MACOSX" in parts or parts[-1].startswith("._") or parts[-1] == ".DS_Store":
                continue
            zf.extract(member, target_dir)


def _extract_all(zip_fn: Union[str, os.PathLike], target_dir: Union[str, os.PathLike]) -> None:
    """Extract a zip file, and the zip files that it contains"""
    _extract_zip(zip_fn, target_dir)
    while True:
        inner_zips = [
            os.path.join(root, fn)
            for root, _, files in os.walk(target_dir)
            for fn in files
            if fn.lower().endswith(".zip")
        ]
        if len(inner_zips) == 0:
            break
        for inner_zip in inner_zips:
            _extract_zip(inner_zip, os.path.dirname(inner_zip))
            os.remove(inner_zip)


def check_dataset(
    dataset_dir: Union[str, os.PathLike],
    zip_name: str = "Dataset.zip",
    labels_dir: Optional[Union[str, os.PathLike]] = None,
) -> None:
    """
    Make sure that a dataset is in the folder `dataset_dir`.

    If the folder does not exist (or is empty), the dataset is set up from the zip file that
    you can download from the course page (`zip_name`): the zip file is extracted (including the
    zip files that it contains), and the folder of the dataset is moved to `dataset_dir`.

    The zip file is searched in the folder with the datasets (the parent of `dataset_dir`), in
    the folder of the notebook (its parent) and in the current folder. In Google Colab, if the
    zip file is not found, you are asked to upload it.

    Parameters
    ----------
    dataset_dir : path
        Folder of the dataset. Its name is the name of the dataset (e.g., "scores_composer_identification")
    zip_name : str
        Name of the zip file with the dataset.
    labels_dir : path (optional)
        If given, the csv files of the zip file (e.g., the labels) are copied to this folder
        (if they are not already there).
    """
    dataset_dir = os.path.abspath(dataset_dir)
    if os.path.isdir(dataset_dir) and len(os.listdir(dataset_dir)) > 0:
        return

    name = os.path.basename(dataset_dir)
    datasets_dir = os.path.dirname(dataset_dir)
    notebook_dir = os.path.dirname(datasets_dir)

    # Look for the zip file (the name can have a suffix, e.g., "Dataset (1).zip" when it is downloaded twice)
    zip_pattern = os.path.splitext(zip_name)[0] + "*.zip"
    zip_fn = None
    for folder in (datasets_dir, notebook_dir, os.getcwd()):
        candidates = glob.glob(os.path.join(folder, zip_pattern))
        if len(candidates) > 0:
            zip_fn = max(candidates, key=os.path.getmtime)
            break

    if zip_fn is None:
        try:
            from google.colab import files  # only available in Google Colab

            print(f"The dataset '{name}' was not found. Please upload the file {zip_name}:")
            uploaded = files.upload()
            uploaded_zips = [fn for fn in uploaded if fn.lower().endswith(".zip")]
            if len(uploaded_zips) > 0:
                zip_fn = os.path.abspath(uploaded_zips[0])
        except ImportError:
            pass

    if zip_fn is None:
        raise FileNotFoundError(
            f"The dataset '{name}' was not found in {dataset_dir}. Download the file {zip_name} "
            f"from the course page, put it in {notebook_dir}, and run this cell again."
        )

    print(f"Setting up the dataset '{name}' from {zip_fn}")
    with tempfile.TemporaryDirectory() as tmp_dir:
        _extract_all(zip_fn, tmp_dir)

        # Find the folder of the dataset: the folder with its name, or the one with the most files
        folders = [root for root, _, _ in os.walk(tmp_dir) if root != tmp_dir]
        named_folders = [folder for folder in folders if os.path.basename(folder) == name]
        if len(named_folders) > 0:
            source_dir = named_folders[0]
        elif len(folders) > 0:
            source_dir = max(folders, key=lambda folder: len(os.listdir(folder)))
        else:
            source_dir = tmp_dir

        if labels_dir is not None:
            os.makedirs(labels_dir, exist_ok=True)
            for root, _, files in os.walk(tmp_dir):
                if root.startswith(source_dir) and source_dir != tmp_dir:
                    continue
                for fn in files:
                    if fn.lower().endswith(".csv") and not os.path.exists(os.path.join(labels_dir, fn)):
                        shutil.copy(os.path.join(root, fn), os.path.join(labels_dir, fn))

        if os.path.isdir(dataset_dir):
            shutil.rmtree(dataset_dir)
        shutil.copytree(source_dir, dataset_dir)
    print(f"Dataset saved in {dataset_dir}")


def plot_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    title: str = "Confusion Matrix",
) -> None:
    """
    Plot a confusion matrix using matplotlib.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted labels (1D array).
    targets : np.ndarray
        True labels (1D array).
    class_names : list of str, optional
        List of class names corresponding to label indices.
        If None, class indices are used as labels.
    normalize : bool, optional
        If True, normalize the confusion matrix by true label counts.
    title : str, optional
        Title of the confusion matrix plot.

    Returns
    -------
    None
        Displays the confusion matrix plot.
    """
    # Compute unique classes and create a class index mapping
    classes = np.unique(np.concatenate((targets, predictions)))
    num_classes = len(classes)
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}

    # Initialize the confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)

    # Populate the confusion matrix
    for t, p in zip(targets, predictions):
        i = class_to_index[t]
        j = class_to_index[p]
        cm[i, j] += 1

    # Normalize the confusion matrix if required
    if normalize:
        with np.errstate(all="ignore"):
            cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)  # Replace NaNs with zeros

    # Use class names if provided
    if class_names is None:
        class_names = [str(cls) for cls in classes]
    else:
        if len(class_names) != num_classes:
            raise ValueError("Length of class_names must match number of classes")

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.Blues
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # Set labels
    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title=title,
    )

    # Rotate the tick labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate each cell with the numeric value
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.show()
