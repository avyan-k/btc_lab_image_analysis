import loading_data as ld
import utils
import doctest
import random
import torch
import pickle
from torchvision import transforms
from torchvision.utils import save_image
import cv2
from pathlib import Path
import os
from tqdm import tqdm
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt    
from matplotlib.backends.backend_pdf import PdfPages
from skimage.metrics import structural_similarity as ssim
# from pytorch_msssim import ssim as torchssim
# from torchmetrics.image import StructuralSimilarityIndexMeasure
from torch_staintools.normalizer import NormalizerBuilder
from heapq import nlargest
def find_cases(image_directory):
    paths = [path for path in Path(image_directory).rglob('*.jpg')]
    cases = list(set([ld.get_case(str(path)) for path in tqdm(paths,desc="Case",position=1,leave=False)]))
    # case_dict = {case:sorted([os.path.join(path.parent,path.name[:-6]+path.name[-4:]) for path in paths if case in path.name]) for case in tqdm(cases,desc="Case",position=1,leave=False)}
    case_dict = {case:sorted([os.path.join(path.parent,path.name) for path in paths if case in path.name]) for case in tqdm(cases,desc="Case",position=1,leave=False)}
    print()
    # for path in tqdm(paths):
    #     os.rename(os.path.join(path.parent,path.name),os.path.join(path.parent,path.name[:-6]+path.name[-4:]))
    return case_dict

def get_norm_paths(tumor_type):
    '''
    Creates folders in ./images/{tumor_type}/ for normalized images and returns the paths for the normalized images
    '''
    image_directory = f"./images/{tumor_type}/images"
    normalize_directory = f"./images/{tumor_type}/normalized_images"
    annotations = ['normal', 'undiff', 'well_diff'] if 'DDC_UC' in tumor_type else ['normal', 'tumor']
    for annotations in annotations:
        Path(os.path.join(normalize_directory,annotations)).mkdir(parents=True, exist_ok=True)
    image_paths = [str(path) for path in Path(image_directory).rglob('*.jpg')]
    parent_list = [os.path.normpath(path).split(os.sep) for path in image_paths]
    for parents in parent_list:
        parents[2] = "normalized_images"
    norm_paths = [os.path.join(*parents) for parents in parent_list]
    return norm_paths
def create_case_pdfs(pdf_directory, case_dict, pages_per_case = -1, normalize = False, normalizer = None, normalization_source_path = '', source_type = ''):
    source_file = ''
    pdf_path = os.path.join(pdf_directory,f'sample_cases.pdf')
    if normalize:
        source_file = os.path.split(normalization_source_path)[1][:-4] #extract case name from path
        assert normalizer is not None and os.path.isfile(normalization_source_path) and source_type != ''
        pdf_path = os.path.join(pdf_directory,f'sample_cases_normalized_with_{source_type}_reference.pdf')
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(3, 3))
        plt.suptitle(f"Base Sample Cases", fontsize=8) 
        ax.axis('off')  # Hide axes  
        if normalize:
            plt.suptitle(f"Reference({source_type}): {source_file}.jpg", fontsize=6)
            ax.imshow(Image.open(normalization_source_path))
        pdf.savefig(fig, dpi=300)  # Increase DPI to 300 for better quality
        plt.close(fig)
        for case_id, image_paths in tqdm(case_dict.items(),desc="Case",position=1,leave=False, total=len(case_dict.keys())):

                images_per_page = 10 * 10

                num_pages = len(image_paths) // images_per_page + int(len(image_paths) % images_per_page > 0)
                if pages_per_case > 0:
                    num_pages = pages_per_case
                for page in tqdm(range (num_pages),desc="Page",position=2,leave=False):
                    # Create a new figure for each page
                    fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(8.5, 11))  # standard letter size
                    plt.suptitle(f"Case/Slide {case_id}", fontsize=16)
                    # Flatten the axes array for easy iteration
                    axes = axes.flatten()
                    # Get the images for the current page
                    page_image_paths = image_paths[page * images_per_page:(page + 1) * images_per_page]
                    for i in range(len(page_image_paths)):
                        image_path = page_image_paths[i]
                        ax = axes[i]
                        if normalize:
                            assert normalizer is not None
                            normalizer = normalizer.to(DEVICE)
                            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB).to(DEVICE)
                            img = normalizer.transform(image)
                        else:
                            img = Image.open(image_path)
                        # Load and display the image
                        ax.imshow(img)
                        ax.axis('off')  # Hide axes
                        # Get the base filename and remove the .jpg extension
                        base_filename = os.path.splitext(os.path.basename(image_path))[0] 
                        # Display the filename without the extension in a smaller font
                        ax.set_title(base_filename, fontsize=3)  # Smaller font size

                    for j in range(i + 1, len(axes)): # type: ignore
                        axes[j].axis('off')
                    # Save the current figure to the PDF with high DPI
                    pdf.savefig(fig, dpi=300)  # Increase DPI to 300 for better quality
                    plt.close(fig)
            # print(f"Clusters saved to {pdf_path} as a PDF.")
    return
def postprocess(image_tensor): 
    return transforms.functional.convert_image_dtype(image_tensor, torch.uint8).squeeze().detach().cpu().permute(1, 2, 0).numpy()
def postprocess_batch(image_tensor):
    return transforms.functional.convert_image_dtype(image_tensor, torch.uint8).detach().cpu().permute(0, 2, 3, 1).numpy()
def postprocess_batch_GPU(image_tensor):
    return transforms.functional.convert_image_dtype(image_tensor, torch.uint8).detach()

def get_fitted_macenko(source_path,seed):
    normalization_source = cv2.cvtColor(cv2.imread(source_path), cv2.COLOR_BGR2RGB)
    source_tensor = transforms.ToTensor()(normalization_source).unsqueeze(0)
    normalizer_macenko = NormalizerBuilder.build('macenko', use_cache=True, concentration_method='ls',rng = seed)
    normalizer_macenko.fit(source_tensor)
    return normalizer_macenko

# def batch_ssim(original_batch, transformed_batch):
#     assert len(original_batch) == len(transformed_batch)
#     original_batch, transformed_batch = postprocess_batch_GPU(original_batch), postprocess_batch_GPU(transformed_batch)
#     return float(torchssim(original_batch, transformed_batch,data_range=1,size_average=True))

def batch_ssim(original_batch, transformed_batch):
    assert len(original_batch) == len(transformed_batch)
    original_batch,transformed_batch  = postprocess_batch(original_batch), postprocess_batch(transformed_batch)
    total = sum([ssim(im1=original_image,im2=transformed_image,data_range=1,channel_axis = 2) for original_image, transformed_image in zip(original_batch, transformed_batch)])
    return total/len(original_batch)

def norm_ssim_dict(case_dict):
    ssim_dict = {}
    for case,images in tqdm(case_dict.items(),leave=False, desc = "Cases"):
        random_source_path = random.choice(images)
        normalizer = get_fitted_macenko(random_source_path,seed)
        normalizer = normalizer.to(DEVICE)
        ssim_values = []
        with torch.no_grad():
            for images,_ in tqdm(image_loader,leave=False, desc = "Images"):
                images = images.to(DEVICE)
                transformed = normalizer.transform(images)
                ssim_values.append(batch_ssim(images,transformed))
        ssim_dict[case] = (sum(ssim_values)/len(ssim_values),random_source_path)
    return ssim_dict
if __name__ == "__main__":
    tumor_type = "SCCOHT_1"
    page_sampled_per_case = 2
    seed = 99
    utils.set_seed(99)
    DEVICE = utils.load_device(99)
    sample_size = 1000
    batch_size = 50
    image_directory = f"./images/{tumor_type}/images"
    result_directory = f"./results/Cases/{tumor_type}"
    Path(result_directory).mkdir(parents=True, exist_ok=True)

    image_loader,_,_  = ld.load_data(batch_size,tumor_type,transforms=transforms.ToTensor(),sample=True,sample_size=sample_size)

    case_dict = find_cases(image_directory)
    if not os.path.isfile(os.path.join(result_directory,f'sample_cases.pdf')):
        create_case_pdfs(result_directory, case_dict, pages_per_case=2)

    
    # ssim_dict ={'15157844_D4': (0.9175280209100558, 'images/SCCOHT_1/images/tumor/15157844_D4_17465st.jpg'), '09K17671C_J0903901': (0.9351928829423319, 'images/SCCOHT_1/images/tumor/09K17671C_J0903901_11676st.jpg'), 'S1614552_2F': (0.8376049164583728, 'images/SCCOHT_1/images/tumor/S1614552_2F_98456st.jpg'), '5063936_D16_J0608633': (0.9311405701924234, 'images/SCCOHT_1/images/tumor/5063936_D16_J0608633_46528st.jpg'), 'B4160_A9': (0.9278923194463178, 'images/SCCOHT_1/images/tumor/B4160_A9_67504st.jpg'), '9829_A9': (0.8314976816275015, 'images/SCCOHT_1/images/tumor/9829_A9_55867st.jpg'), 'B4163_A6': (0.9369949175371467, 'images/SCCOHT_1/images/tumor/B4163_A6_75580st.jpg'), '524996_A4': (0.9367939456971854, 'images/SCCOHT_1/images/tumor/524996_A4_47471st.jpg'), '1221745': (0.918558993686121, 'images/SCCOHT_1/images/tumor/1221745_14575st.jpg'), '44020640B': (0.8204935151511342, 'images/SCCOHT_1/images/tumor/44020640B_38316st.jpg'), 'BM16108759': (0.9137875374761252, 'images/SCCOHT_1/images/tumor/BM16108759_77476st.jpg'), 'SF1925758_B6': (0.9005475439141526, 'images/SCCOHT_1/images/tumor/SF1925758_B6_102440st.jpg'), '473M': (0.9117230326317429, 'images/SCCOHT_1/images/tumor/473M_45094st.jpg'), 'S029532_A7': (0.8627600334003198, 'images/SCCOHT_1/images/tumor/S029532_A7_93860st.jpg'), '588695_14': (0.9346372538581823, 'images/SCCOHT_1/images/tumor/588695_14_49075st.jpg'), '1060514': (0.9232577027696263, 'images/SCCOHT_1/images/tumor/1060514_12426st.jpg'), 'B4162_A7': (0.8982288545105136, 'images/SCCOHT_1/images/tumor/B4162_A7_72704st.jpg'), '133284A': (0.8893258325396313, 'images/SCCOHT_1/images/tumor/133284A_15892st.jpg'), '334811112': (0.9042949459426952, 'images/SCCOHT_1/images/tumor/334811112_36088st.jpg'), 'Q121348773B': (0.9256444643407311, 'images/SCCOHT_1/images/tumor/Q121348773B_93061st.jpg'), '15D16367_D3': (0.8685215883908683, 'images/SCCOHT_1/images/tumor/15D16367_D3_18970st.jpg'), 'H1521467_B3': (0.9104381847060599, 'images/SCCOHT_1/images/tumor/H1521467_B3_84876st.jpg'), 'tumor65_1063389': (0.9023529538072416, 'images/SCCOHT_1/images/tumor/tumor65_1063389_104488st.jpg'), '20B0004537_A1': (0.9054150175860524, 'images/SCCOHT_1/images/tumor/20B0004537_A1_26084st.jpg'), '06330054_A1': (0.883218569318074, 'images/SCCOHT_1/images/tumor/06330054_A1_7947st.jpg'), '49020640': (0.8503109155130153, 'images/SCCOHT_1/images/tumor/49020640_46226st.jpg'), 'S173033_A17_SC11': (0.8782415822493214, 'images/SCCOHT_1/images/tumor/S173033_A17_SC11_99954st.jpg'), 'AS21032651_D1': (0.932088032014127, 'images/SCCOHT_1/images/tumor/AS21032651_D1_63689st.jpg'), 'H107043': (0.9020396527302864, 'images/SCCOHT_1/images/tumor/H107043_79496st.jpg'), '21509636_A4': (0.9029030210438126, 'images/SCCOHT_1/images/tumor/21509636_A4_27921st.jpg'), 'B4161_C1': (0.9238022862057256, 'images/SCCOHT_1/images/tumor/B4161_C1_69703st.jpg'), 'OC1813435_A4': (0.8017241022753715, 'images/SCCOHT_1/images/tumor/OC1813435_A4_87349st.jpg'), '22B33007_A3': (0.905060368332009, 'images/SCCOHT_1/images/tumor/22B33007_A3_34165st.jpg'), 'GS986_1B': (0.9112375266043811, 'images/SCCOHT_1/images/tumor/GS986_1B_78269st.jpg'), '0079682': (0.8983942788972699, 'images/SCCOHT_1/images/tumor/0079682_6232st.jpg'), '18292_T5': (0.9378091358369564, 'images/SCCOHT_1/images/tumor/18292_T5_22900st.jpg'), '20B0014943_A6': (0.9091713970525122, 'images/SCCOHT_1/images/tumor/20B0014943_A6_26432st.jpg'), '595432120': (0.923726274958486, 'images/SCCOHT_1/images/tumor/595432120_49936st.jpg'), 'P1542230_1E': (0.9182550062678856, 'images/SCCOHT_1/images/tumor/P1542230_1E_91281st.jpg'), '22B12341_A1': (0.7875227667867535, 'images/SCCOHT_1/images/tumor/22B12341_A1_30847st.jpg'), 'B1405099': (0.9186457767613461, 'images/SCCOHT_1/images/tumor/B1405099_64386st.jpg'), 'AS17036383_A49': (0.9080613729976141, 'images/SCCOHT_1/images/tumor/AS17036383_A49_62174st.jpg'), 'H16002710_1B': (0.9015840982135792, 'images/SCCOHT_1/images/tumor/H16002710_1B_85818st.jpg'), '2001B1486_PCS3208': (0.9127805683631813, 'images/SCCOHT_1/images/tumor/2001B1486_PCS3208_25637st.jpg'), '881175B4': (0.9216726983239947, 'images/SCCOHT_1/images/tumor/881175B4_52878st.jpg'), 'PCS281092299G': (0.9145045450380239, 'images/SCCOHT_1/images/normal/PCS281092299G_5116sn.jpg')}
    ssim_dict = norm_ssim_dict(case_dict)
    print(ssim_dict)
    best_norm_cases = nlargest(3, ssim_dict, key=lambda k: ssim_dict[k][0])
    print(*[ssim_dict[case] for case in best_norm_cases])
    with open(f"./pickle/ssim_{seed}_{sample_size}",'wb') as f:
        pickle.dump(obj=ssim_dict,file=f,protocol=pickle.HIGHEST_PROTOCOL)

    for case in best_norm_cases:
        source_path = ssim_dict[case][1]
        image = cv2.cvtColor(cv2.imread(source_path), cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.show()

    p = ['images/SCCOHT_1/images/tumor/18292_T5_23763st.jpg', 'images/SCCOHT_1/images/tumor/B4161_C1_69288st.jpg','images/SCCOHT_1/images/tumor/595432120_52009st.jpg']  
    reference_slide_path = "./images/SCCOHT_1/images/normal/15D16367_D3_247sn.jpg"











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
    #     normalization_source = cv2.cvtColor(cv2.imread(reference_slide_path), cv2.COLOR_BGR2RGB)
    #     normalizer.fit(T(normalization_source))

    #     if not os.path.isfile(os.path.join(result_directory,f'sample_cases_normalized_with_{reference_type}_reference.pdf')):
    #         create_case_pdfs(result_directory,case_dict,pages_per_case=page_sampled_per_case,normalize = True, normalizer=normalizer,normalization_source_path=reference_slide_path,source_type=reference_type)
    

    

    
    
    # paths = [str(path) for path in Path(image_directory).rglob('*.jpg')][:50]
    
    
    
    
    # img = stain_normalize(os.path.join(image_directory,'tumor','S173033_A17_SC11_100959st.jpg'), normalizer)
    # img.show()
    # create_case_pdfs(result_directory,case_df,normalize = False)

    
    # 
    

    # norms = stain_normalize(f'images/SCCOHT_1/images/normal/15D16367_D3_247sn.jpg', paths, normalizer)
