import loading_data as ld
import utils
import random
import torch
from torch.linalg import LinAlgError
import pickle
from torchvision import transforms, io
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import cv2 as cv
from pathlib import Path
import os
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from piqa import SSIM, PSNR

# from torchmetrics import structural_similarity_index_measure as ssim
# from pytorch_msssim import ssim as torchssim
# from torchmetrics.image import StructuralSimilarityIndexMeasure
from torch_staintools.normalizer import NormalizerBuilder
from torch_staintools.functional.tissue_mask import TissueMaskException


def get_norm_paths(tumor_type):
    """
    Creates folders in ./images/{tumor_type}/ for normalized images and returns the paths for the normalized images
    """
    image_directory = f"./images/{tumor_type}/images"
    normalize_directory = f"./images/{tumor_type}/normalized_images"
    annotations = (
        ["normal", "undiff", "well_diff"]
        if "DDC_UC" in tumor_type
        else ["normal", "tumor"]
    )
    for annotations in annotations:
        Path(os.path.join(normalize_directory, annotations)).mkdir(
            parents=True, exist_ok=True
        )
    image_paths = [str(path) for path in Path(image_directory).rglob("*.jpg")]
    parent_list = [os.path.normpath(path).split(os.sep) for path in image_paths]
    for parents in parent_list:
        parents[2] = "normalized_images"
    norm_paths = [os.path.join(*parents) for parents in parent_list]
    return norm_paths


def create_case_pdfs(
    pdf_directory,
    case_dict,
    pages_per_case=-1,
    normalize=False,
    normalizer=None,
    normalization_source_path="",
    source_type="",
):
    source_file = ""
    pdf_path = os.path.join(pdf_directory, "sample_cases.pdf")
    T = transforms.ToTensor()
    if normalize:
        source_file = os.path.split(normalization_source_path)[1][
            :-4
        ]  # extract case name from path
        assert (
            normalizer is not None
            and os.path.isfile(normalization_source_path)
            and source_type != ""
        )
        pdf_path = os.path.join(
            pdf_directory, f"sample_cases_normalized_with_{source_type}_reference.pdf"
        )
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(3, 3))
        plt.suptitle("Base Sample Cases", fontsize=8)
        ax.axis("off")  # Hide axes
        if normalize:
            plt.suptitle(f"Reference({source_type}): {source_file}.jpg", fontsize=6)
            ax.imshow(Image.open(normalization_source_path))  # type: ignore
        pdf.savefig(fig, dpi=300)  # Increase DPI to 300 for better quality
        plt.close(fig)
        for case_id, image_paths in tqdm(
            case_dict.items(),
            desc="Case",
            position=1,
            leave=False,
            total=len(case_dict.keys()),
        ):
            images_per_page = 10 * 10

            num_pages = len(image_paths) // images_per_page + int(
                len(image_paths) % images_per_page > 0
            )
            if pages_per_case > 0:
                num_pages = pages_per_case
            for page in tqdm(range(num_pages), desc="Page", position=2, leave=False):
                # Create a new figure for each page
                fig, axes = plt.subplots(
                    nrows=10, ncols=10, figsize=(8.5, 11)
                )  # standard letter size
                plt.suptitle(f"Case/Slide {case_id}", fontsize=16)
                # Flatten the axes array for easy iteration
                axes = axes.flatten()
                # Get the images for the current page
                page_image_paths = image_paths[
                    page * images_per_page : (page + 1) * images_per_page
                ]
                for i in range(len(page_image_paths)):
                    image_path = page_image_paths[i]
                    ax = axes[i]
                    if normalize:
                        assert normalizer is not None
                        normalizer = normalizer.to(DEVICE)
                        image = T(
                            cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB)
                        )
                        img = normalizer.transform(image)
                    else:
                        img = Image.open(image_path)
                    # Load and display the image
                    ax.imshow(img)
                    ax.axis("off")  # Hide axes
                    # Get the base filename and remove the .jpg extension
                    base_filename = os.path.splitext(os.path.basename(image_path))[0]
                    # Display the filename without the extension in a smaller font
                    ax.set_title(base_filename, fontsize=3)  # Smaller font size

                for j in range(i + 1, len(axes)):  # type: ignore
                    axes[j].axis("off")
                # Save the current figure to the PDF with high DPI
                pdf.savefig(fig, dpi=300)  # Increase DPI to 300 for better quality
                plt.close(fig)
        # print(f"Clusters saved to {pdf_path} as a PDF.")
    return


def postprocess(image_tensor):
    return (
        transforms.functional.convert_image_dtype(image_tensor, torch.uint8)
        .squeeze()
        .detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )  # type: ignore


def postprocess_batch(image_tensor):
    return (
        transforms.functional.convert_image_dtype(image_tensor, torch.uint8)
        .detach()
        .cpu()
        .permute(0, 2, 3, 1)
        .numpy()
    )  # type: ignore


def postprocess_batch_GPU(image_tensor):
    return transforms.functional.convert_image_dtype(image_tensor, torch.uint8).detach()  # type: ignore


def get_fitted_macenko(source_path, seed):
    normalization_source = cv.cvtColor(cv.imread(source_path), cv.COLOR_BGR2RGB)
    source_tensor = transforms.ToTensor()(normalization_source).unsqueeze(0)
    normalizer_macenko = NormalizerBuilder.build(
        "macenko", use_cache=True, concentration_method="ls", rng=seed
    )
    normalizer_macenko.fit(source_tensor)
    return normalizer_macenko


def batch_metrics(original_batch, transformed_batch):
    assert len(original_batch) == len(transformed_batch)
    # original_batch,transformed_batch  = postprocess_batch(original_batch), postprocess_batch(transformed_batch)
    # total = sum([ssim(im1=original_image,im2=transformed_image,data_range=1,channel_axis = 2) for original_image, transformed_image in zip(original_batch, transformed_batch)])
    ssim = SSIM().to(DEVICE)
    psnr = PSNR().to(DEVICE)
    return torch.Tensor(
        [
            ssim(original_batch, transformed_batch),
            psnr(original_batch, transformed_batch),
        ]
    )


def normalization_evaluation(tumor_type, seed, images_per_case, sample_size):
    metric_dict = {}
    image_directory = f"./images/{tumor_type}/images"
    result_directory = f"./results/Cases/{tumor_type}"
    Path("./pickle").mkdir(parents=True, exist_ok=True)
    Path(result_directory).mkdir(parents=True, exist_ok=True)

    case_dict = ld.find_cases(image_directory)
    if not os.path.isfile(os.path.join(result_directory, "sample_cases.pdf")):
        create_case_pdfs(result_directory, case_dict, pages_per_case=1)

    pickle_path = f"./pickle/metrics_{tumor_type}_{seed}_{sample_size}_samples_{images_per_case}_images.pkl"
    if os.path.isfile(pickle_path):
        with open(pickle_path, "rb") as f:
            try:
                metric_dict = pickle.load(f)
            except EOFError:
                pass
    if (
        len(metric_dict.keys()) == len(case_dict.keys()) * images_per_case
    ):  # we sample image for every case, and if metric_dict has that many keys, then  went through entire dataset
        return metric_dict

    if sample_size == "all":
        sample_size = ld.get_size_of_dataset(image_directory, extension="jpg")
    image_loader, _, _, _ = ld.load_data(
        300,
        image_directory,
        transforms=transforms.ToTensor(),
        sample=True,
        sample_size=sample_size,
    )
    for case, images in tqdm(case_dict.items(), leave=False, desc="Cases"):
        random_source_paths = random.sample(images, images_per_case)
        for random_source_path in tqdm(
            random_source_paths, leave=False, desc="Sources"
        ):
            normalizer = get_fitted_macenko(random_source_path, seed)
            normalizer = normalizer.to(DEVICE)
            metrics_values = torch.zeros(2)
            with torch.no_grad():
                for images, _ in tqdm(image_loader, leave=False, desc="Images"):
                    images = images.to(DEVICE)
                    transformed = normalizer.transform(images)
                    metrics_values += batch_metrics(images, transformed)
            metric_dict[random_source_path] = metrics_values.cpu() / max(
                1, sample_size / 300
            )
            with open(pickle_path, "wb") as f:
                pickle.dump(obj=metric_dict, file=f, protocol=pickle.HIGHEST_PROTOCOL)
    return metric_dict


def save_normalized_images(tumor_type, source_path, seed):
    batch_size = 300
    image_dataset = ld.get_image_dataset(
        tumor_type=tumor_type,seed=seed,normalized=False
    )
    filepaths = list(zip(*image_dataset.samples))[0]
    image_loader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=ld.get_allowed_forks(),
    )
    for label in set(image_dataset.classes):
        Path(f"./images/{tumor_type}/normalized_images/{label}").mkdir(
            parents=True, exist_ok=True
        )
    filepath_iterator = iter(filepaths)
    normalizer = get_fitted_macenko(source_path, seed)
    normalizer = normalizer.to(DEVICE)

    for batch_index, (images, _) in enumerate(tqdm(image_loader, leave=False)):
        images = images.to(DEVICE)
        try:
            normalizer.transform(images)
        except (TissueMaskException, LinAlgError, IndexError):
            unnormalizable_filepaths = filepaths[
                batch_index * batch_size : min(
                    len(filepaths), (batch_index + 1) * batch_size
                )
                - 1
            ]
            check_unnormalizable_images(
                filepaths=unnormalizable_filepaths, source_path=source_path, seed=seed
            )  # log unnormalizable images
            for image in images:  # try normalizing each image individually
                try:
                    normalizer.transform(image.unsqueeze(0))
                except (
                    TissueMaskException,
                    LinAlgError,
                    IndexError,
                ):  # these three exceptions are already logged, so we skip them
                    continue
                filepath = Path(next(filepath_iterator))
                normalized_filepath = utils.rename_dir(filepath, 2, "normalized_images")
                save_image(image, normalized_filepath)
            continue
        except Exception as e:
            raise e
        for image in images:
            filepath = Path(next(filepath_iterator))
            # print(filepath)
            normalized_filepath = utils.rename_dir(filepath, 2, "normalized_images")
            # print(normalized_filepath)
            save_image(image, normalized_filepath)
        break


def check_unnormalizable_images(filepaths, source_path, seed):
    normalizer = get_fitted_macenko(source_path, seed)
    normalizer = normalizer.to(DEVICE)

    with open(file=f"./results/{tumor_type}_unnormalizable_images.txt", mode="a") as f:
        f.write(f"Checked on {utils.get_time()}\n")
        for filepath in tqdm(filepaths, leave=False):
            image = io.read_image(filepath)
            image = image.unsqueeze(0).to(DEVICE)
            try:
                normalizer.transform(image)
            except (TissueMaskException, LinAlgError, IndexError) as err:
                f.write(f"{filepath} {err}\n")
            except Exception as e:  # any other is not logged so raise exception
                raise e


if __name__ == "__main__":
    seed = 99
    utils.set_seed(99)
    DEVICE = utils.load_device(99)
    page_sampled_per_case = 2
    sample_size = 15000
    images_per_case = 3

    best_sources = {
        "vMRT": "./images/vMRT/images/tumor/AC23029182_A1_1_1881.jpg",
        "SCCOHT_1": "./images/SCCOHT_1/images/tumor/AS21032651_D1_63597.jpg",
        "DDC_UC_1": "./images/DDC_UC_1/images/normal/AS15043088_62194.jpg",
    }

    for tumor_type in os.listdir("./images"):
        print(tumor_type)
        if tumor_type in [".DS_Store", "__MACOSX"]:
            continue
        # check_unnormalizable_images(tumor_type,best_sources[tumor_type],seed)
        # with open(file=f"./results/{tumor_type}_unnormalizable_images.txt",mode='r') as f:
        #     f.readline()
        #     for line in f:
        #         to_delete = line.split()[0]
        #         if os.path.isfile(to_delete):
        #             os.remove(to_delete)
        save_normalized_images(
            tumor_type=tumor_type, source_path=best_sources[tumor_type], seed=seed
        )
        break
        # metric_dict = normalization_evaluation(tumor_type=tumor_type,seed= seed,images_per_case=images_per_case, sample_size = sample_size)
        # best_norm_cases = nlargest(3, metric_dict.items(), key = lambda k : k[0][1]) # type: ignore
        # with open(f"./pickle/metrics_{tumor_type}_best_norm_cases.txt",'w') as f:
        #         for case, metrics in best_norm_cases:
        #             # f.write(f"Case: {case} SSIM: {float(metrics[0])} PSNR: {float(metrics[1])}\n")
        #             f.write(f"Case: {case} SSIM: {float(metrics)}\n")
        # print(*best_norm_cases)

    # p = ['images/SCCOHT_1/images/tumor/18292_T5_23763st.jpg', 'images/SCCOHT_1/images/tumor/B4161_C1_69288st.jpg','images/SCCOHT_1/images/tumor/595432120_52009st.jpg']
    # reference_slide_path = "./images/SCCOHT_1/images/normal/15D16367_D3_247sn.jpg"


#     image_paths = [r'images\SCCOHT_1\images\tumor\AS17036383_A49_58057st.jpg',
# r'images\SCCOHT_1\images\tumor\PCS281092299G_92675st.jpg']
#     norm_paths = [r'images\SCCOHT_1\normalized_images\tumor\AS17036383_A49_58057st.jpg',
# r'images\SCCOHT_1\normalized_images\tumor\PCS281092299G_92675st.jpg']
# with open(file=f"./results/{tumor_type}_unnormalizable_images.txt",mode='w') as f:
#     f.write(f"Checked on {utils.get_time()}\n")
#     for i,path in enumerate(tqdm(image_paths)):
#         if not os.path.isfile(norm_paths[i]):
#             try:
#                 stain_normalize(path,normalizer,save=True, save_path = norm_paths[i])
#             except:
#                 f.write(str(path)+'\n')

# reference_slide_path = "./images/SCCOHT_1/images/normal/15D16367_D3_247sn.jpg"

# normalization_dictionnary = {
#     'SCCOHT_1' : ({
#         'bright' : '15D16367_D3_247sn.jpg',
#         'dark' : 'GS986_1B_78169st.jpg',
#         'indigo' : '22B33007_A3_34208st.jpg'
#     },['Q121348773B','AS21032651_D1','0079682','H107043','22B33007_A3','GS986_1B','524996_A4']),
#     'vMRT' : ({
#         'dark' : 'AC23029182_A1_1_52vn.jpg'
#     },[]),
#     'DDC_UC_1' : ({

#     },[])
# }

# case_dict = find_cases(image_directory)
# create_case_pdfs_by_case(result_directory,case_dict,normalize = False)

# references, samples_cases = normalization_dictionnary[tumor_type]
# case_dict = find_cases(image_directory)

# normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
# T = transforms.Compose([
# transforms.ToTensor(),
# transforms.Resize((256,256),antialias="True"),
# transforms.Lambda(lambda x: x*255)
# ])
# if not os.path.isfile(os.path.join(result_directory,f'sample_cases.pdf')):
#     create_case_pdfs(result_directory,case_dict,pages_per_case=page_sampled_per_case,normalize = False)
# for reference_type, reference_slide in references.items():
#     annotation_prefix = reference_slide.split('.')[0][-1]
#     if tumor_type in "DDC_UC":
#         assert annotation_prefix in 'nuw' # annotation is either normal, undifferentiated or well-differentiated
#         match annotation_prefix:
#             case 'n': annotation = 'normal'
#             case 'u': annotation = 'undiff'
#             case 'w': annotation = 'well_diff'
#     else:
#         assert annotation_prefix in 'nt' # annotation is either tumor or normal
#         annotation = 'normal' if annotation_prefix == 'n' else 'tumor'
#     reference_slide_path = os.path.join(image_directory,annotation, reference_slide) # type: ignore
#     # print(reference_slide_path)
#     normalization_source = cv.cvtColor(cv.imread(reference_slide_path), cv.COLOR_BGR2RGB)
#     normalizer.fit(T(normalization_source))

#     if not os.path.isfile(os.path.join(result_directory,f'sample_cases_normalized_with_{reference_type}_reference.pdf')):
#         create_case_pdfs(result_directory,case_dict,pages_per_case=page_sampled_per_case,normalize = True, normalizer=normalizer,normalization_source_path=reference_slide_path,source_type=reference_type)


# paths = [str(path) for path in Path(image_directory).rglob('*.jpg')][:50]


# img = stain_normalize(os.path.join(image_directory,'tumor','S173033_A17_SC11_100959st.jpg'), normalizer)
# img.show()
# create_case_pdfs(result_directory,case_df,normalize = False)


#


# norms = stain_normalize(f'images/SCCOHT_1/images/normal/15D16367_D3_247sn.jpg', paths, normalizer)
