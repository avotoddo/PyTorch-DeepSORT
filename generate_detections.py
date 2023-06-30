
import os
import errno
import argparse
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms

def extract_image_patch(image, bbox, patch_shape):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    return image


def generate_detections(encoder, mot_dir, output_name, output_dir, detection_dir=None):
    """Generate detections with features.

    Parameters
    ----------
    mot_dir : str
        Path to the MOTChallenge directory (can be either train or test).
    output_dir
        Path to the output directory. Will be created if it does not exist.
    detection_dir
        Path to custom detections. The directory structure should be the default
        MOTChallenge structure: `[sequence]/det/det.txt`. If None, uses the
        standard MOTChallenge detections.

    """
    if detection_dir is None:
        detection_dir = mot_dir
    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno == errno.EEXIST and os.path.isdir(output_dir):
            pass
        else:
            raise ValueError(
                "Failed to created output directory '%s'" % output_dir)

                
    transform= transforms.Compose([ \
			  transforms.ToPILImage(),\
				transforms.Resize((128,128)),\
			  transforms.ToTensor()]) # defining transform outside of loop to save space

    net = torch.load(encoder).to(device = 'cuda') #defining net outside of loop
    net = net.eval()
    sequence_dir = mot_dir


    image_dir = sequence_dir
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in   os.listdir(image_dir)}

    detection_file = detection_dir
    detections_in = np.loadtxt(detection_file, delimiter=',')
    detections_out = []

    frame_indices = detections_in[:, 0].astype(int)
    min_frame_idx = frame_indices.astype(int).min()
    max_frame_idx = frame_indices.astype(int).max()
    for frame_idx in range(min_frame_idx, max_frame_idx + 1):
        print("Frame %05d/%05d" % (frame_idx, max_frame_idx))
        mask = frame_indices == frame_idx
        rows = detections_in[mask]

        if frame_idx not in image_filenames:
            print("WARNING could not find image for frame %d" % frame_idx)
            continue
        bgr_image = cv2.imread(
            image_filenames[frame_idx], cv2.IMREAD_COLOR)

        # features = encoder(bgr_image, rows[:, 2:6].copy()) This is the tensor flow line of code that uses an encoder object
        # essentially, it is going through every bounding box, and encoding the information into a
        features = []
        for row in rows:
          patch = extract_image_patch(bgr_image, row[2:6].copy(), None) # gets region of image in bbox
          patch = transform(patch).to(device = 'cuda') # converting image to tensor
          patch = torch.unsqueeze(patch,0)
          features.append(net.forward_once(patch).detach().cpu().numpy()) # runs that area of image through net and then appends it to features list
        det = [np.r_[(row, feature)] for row, feature
                          in zip(rows, features)]
        detections_out += det

    output_filename = os.path.join(output_dir, "%s.npy" %output_name )
    np.save(
        output_filename, np.asarray(detections_out), allow_pickle=False)


def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Re-ID feature extractor")
    parser.add_argument(
        "--encoder", help = "Path to net for encoding images",
        required = True)
    parser.add_argument(
        "--mot_dir", help="Path to MOTChallenge directory (train or test)",
        required=True)
    parser.add_argument(
        "--output_name", help = 'filename for output',
        required = True)
    parser.add_argument(
        "--detection_dir", help="Path to custom detections. Defaults to "
        "standard MOT detections Directory structure should be the default "
        "MOTChallenge structure: [sequence]/det/det.txt", default=None)
    parser.add_argument(
        "--output_dir", help="Output directory. Will be created if it does not"
        " exist.", default="detections")
    return parser.parse_args()


def main():
    args = parse_args()
    generate_detections(args.encoder, args.mot_dir, args.output_name, args.output_dir,
                        args.detection_dir)


if __name__ == "__main__":
    main()