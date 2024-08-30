import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def save_clusters_to_pdf(cluster_info, pdf_output_path):
    """
    Creates a pdf files displaying the image tiles that belong to each k-means cluster
        cluster_info: pandas df with col 'ImagePath' and 'Cluster'

    """
    print("\nCreating PDF with image tiles ...")
    
    # Create a PdfPages object
    with PdfPages(pdf_output_path) as pdf:
        # Get unique clusters from 'Cluster' column in df
        unique_clusters = sorted(cluster_info['Cluster'].unique())
        
        for cluster in unique_clusters:
            # Get the images in the current cluster
            cluster_images_info = cluster_info[cluster_info['Cluster'] == cluster] # subset of cluster_info
            image_paths = cluster_images_info['ImagePath'].tolist() # paths for the images in the cluster

            # Calculate the number of pages needed for this cluster
            images_per_page = 100  # 10 x 10 grid
            num_pages = len(image_paths) // images_per_page + int(len(image_paths) % images_per_page > 0)
            
            for page in range(num_pages):
                # Create a new figure for each page
                fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(8.5, 11))  # standard letter size
                plt.suptitle(f"Cluster {cluster}", fontsize=16)
                
                # Flatten the axes array for easy iteration
                axes = axes.flatten()

                # Get the images for the current page
                page_image_paths = image_paths[page * images_per_page:(page + 1) * images_per_page]

                for i, (ax, image_path) in enumerate(zip(axes, page_image_paths)):
                    # Adjust the file path to remove any redundant directory parts
                    # image_path = image_path.replace(base_image_directory, "").lstrip("/")
                    # full_image_path = os.path.join(base_image_directory, image_path)
                    
                    # Load and display the image
                    # img = Image.open(full_image_path)
                    img = Image.open(image_path)
                    ax.imshow(img)
                    ax.axis('off')  # Hide axes
                    
                    # Get the base filename and remove the .jpg extension
                    base_filename = os.path.splitext(os.path.basename(image_path))[0]
                    
                    # Display the filename without the extension in a smaller font
                    ax.set_title(base_filename, fontsize=3)  # Smaller font size
                
                # Hide any remaining empty subplots
                for j in range(i + 1, len(axes)):
                    axes[j].axis('off')

                # Save the current figure to the PDF with high DPI
                pdf.savefig(fig, dpi=300)  # Increase DPI to 300 for better quality
                plt.close(fig)

    print(f"Clusters saved to {pdf_output_path} as a PDF.")
    return