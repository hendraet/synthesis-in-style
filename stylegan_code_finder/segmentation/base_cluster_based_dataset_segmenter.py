import copy
import json
import pickle
from collections import defaultdict
from functools import reduce
from itertools import combinations
from typing import List, Dict, Union, Tuple, Set

import cv2
import numpy
import torch

from segmentation.base_dataset_segmenter import BaseDatasetSegmenter
from utils.segmentation_utils import draw_contours_on_same_sized_canvases, ClassContours, PredictedClusters, \
    ClassContoursForSubImages, BBox


class BaseClusterBasedDatasetSegmenter(BaseDatasetSegmenter):
    """
    An abstract superclass for all DatasetSegmenters that rely on clusters that were calculated on the intermediate
    features of a synthesis network (e.g. StyleGAN)
    """
    def __init__(self, *args, keys_for_class_determination: List[str], keys_for_finegrained_segmentation: List[str],
                 num_clusters: int, min_class_contour_area: int, only_keep_overlapping: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.keys_for_class_determination = keys_for_class_determination
        self.keys_for_finegrained_segmentation = keys_for_finegrained_segmentation
        self.keys_for_generation = self.keys_for_class_determination + self.keys_for_finegrained_segmentation
        self.num_clusters = num_clusters
        self.catalog = self.load_catalog()
        self.handwriting_overlap_threshold = 0.5
        self.min_class_contour_area = min_class_contour_area
        self.only_keep_overlapping = only_keep_overlapping
        self.class_label_map = self.load_class_label_map()

    def adjust_catalog(self, catalog: dict) -> dict:
        adjusted_catalog = {}
        for key in catalog.keys():
            if key in self.keys_for_generation:
                adjusted_catalog[key] = catalog[key]
        return adjusted_catalog

    def load_catalog(self) -> dict:
        file_name = self.base_dir / 'catalogs' / f'{self.num_clusters}.pkl'
        with file_name.open('rb') as f:
            try:
                catalogs = pickle.load(f)
            except ModuleNotFoundError:
                # we are loading a legacy catalog -> we need to adjust the module paths
                import sys
                from segmentation import gan_local_edit
                sys.modules['gan_local_edit'] = gan_local_edit
                catalogs = pickle.load(f)
        return self.adjust_catalog(catalogs)

    def load_class_label_map(self) -> Dict[str, Dict[str, list]]:
        map_file_name = self.base_dir / f"merged_classes_{self.num_clusters}.json"
        with map_file_name.open() as f:
            class_label_map = json.load(f)

        inverted_class_label_map = {}
        for key, sub_label_map in class_label_map.items():
            inverted_sub_label_map = defaultdict(list)
            for sub_key, label_name in sub_label_map.items():
                inverted_sub_label_map[label_name].append(int(sub_key))
            inverted_class_label_map[key] = inverted_sub_label_map
        return inverted_class_label_map

    def check_sanity_of_class_label_map(self, relevant_keys: Set) -> Dict:
        color_keys = list(self.class_to_color_map.keys())
        unlabelled_clusters = {}
        for relevant_key in relevant_keys:
            for class_label in self.class_label_map[relevant_key]:
                if class_label not in color_keys:
                    if relevant_key not in unlabelled_clusters:
                        unlabelled_clusters[relevant_key] = []
                    unlabelled_clusters[relevant_key].append(class_label)
        return unlabelled_clusters

    def render_debug_contours(self, contours: ClassContours, name: str):
        debug_images = []
        batch_size = len(list(contours.values())[0])
        for batch_id in range(batch_size):
            debug_image = numpy.zeros((self.image_size + self.max_debug_text_size, self.image_size, 3), dtype='uint8')
            # add white bounds around the image
            debug_image = cv2.rectangle(debug_image, (0, 0), debug_image.shape[:2], color=(255, 255, 255))
            for class_name, class_contour_batch in contours.items():
                if class_name == 'background':
                    continue

                image_contours = class_contour_batch[batch_id]
                color = self.class_to_color_map[class_name]

                if image_contours is None:
                    # there is nothing in this image
                    continue

                contour_image = numpy.zeros_like(debug_image)
                cv2.drawContours(contour_image, image_contours, -1, color=color, thickness=cv2.FILLED)
                debug_image = cv2.addWeighted(debug_image, 0.5, contour_image, 0.5, 1.0)

            font_scale = 1
            font = cv2.FONT_HERSHEY_PLAIN
            text_size, baseline = cv2.getTextSize(name, font, font_scale, 1)
            assert text_size[1] < self.max_debug_text_size, "Debug text is too large!"
            debug_image = cv2.rectangle(
                debug_image,
                [0, self.image_size, self.image_size, self.image_size + self.max_debug_text_size],
                (255, 255, 255),
                cv2.FILLED
            )
            debug_image = cv2.putText(
                debug_image, name, (0, self.image_size + self.max_debug_text_size - baseline), font, font_scale,
                (0, 0, 0)
            )
            debug_images.append(debug_image.copy())
        self.debug_images[name] = copy.copy(debug_images)

    def predict_clusters(self, activations: Dict[int, torch.Tensor], class_label_map: Dict[str, Dict[str, list]]) \
            -> PredictedClusters:
        predicted_classes = {}
        activations = {str(k): v for k, v in activations.items()}
        for layer_id, factor_catalog in self.catalog.items():
            activations_for_layer = activations[layer_id]
            # use k-means cluster centers to predict class membership of each pixel
            class_membership = factor_catalog.predict(activations_for_layer)

            # now merge all classes that belong to the same superclass together into one image with a channel for
            # each class maps from class name to array of shape [batch_size, height, width]
            images_per_class = {}
            for class_name, class_ids in class_label_map[layer_id].items():
                masks = [class_membership == class_id for class_id in class_ids]
                mask = reduce(torch.bitwise_or, masks, torch.zeros_like(masks[0], dtype=torch.bool))

                images_per_class[class_name] = mask

            predicted_classes[layer_id] = images_per_class
        return predicted_classes

    def prepare_image_segmentation(self, activations, class_label_map):
        if self.debug:
            self.debug_images.clear()

        predicted_clusters = self.predict_clusters(activations, class_label_map)
        predicted_clusters = self.resize_to_image_size(predicted_clusters)
        return predicted_clusters

    def cluster_image_to_contours(self, cluster_arrays: numpy.ndarray) -> List[List[numpy.ndarray]]:
        batch_contours = []
        for image in cluster_arrays:
            image = self.dilate_image(image)
            contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            batch_contours.append(contours)
        return batch_contours

    def contour_overlap(self, contour1: numpy.ndarray, contour2: numpy.ndarray) -> int:
        """
        Draws contours on two separate canvases and applies logical and. If the contours are overlapping at least one
        element of the resulting array is True.
        """
        contours = [contour1, contour2]

        bboxes = [BBox(*c.min(axis=0)[0], *c.max(axis=0)[0]) for c in contours]
        assert len(bboxes) == 2
        if not bboxes[0].is_overlapping_with(bboxes[1]):
            # If the bounding boxes of the contours don't overlap the contours won't either and we can omit the
            # expensive drawing procedure
            return 0

        # using minimal_canvas to keep numpy arrays as small as possible
        images = draw_contours_on_same_sized_canvases(contours, minimal_canvas=True)

        overlap = numpy.logical_and(*images)
        values, counts = numpy.unique(overlap, return_counts=True)
        try:
            # Creates a dictionary that maps each unique value (in this case only True or False) to the number of
            # times it occurs. Getting the count for True corresponds to the number of overlapping pixels.
            # No overlap results in a KeyError
            return dict(zip(values, counts))[True]
        except KeyError:
            return 0

    def merge_two_contours_if_overlapping(self, contour1: numpy.ndarray, contour2: numpy.ndarray) \
            -> Union[List[numpy.ndarray], None]:
        if self.contour_overlap(contour1, contour2) > 0:
            images = draw_contours_on_same_sized_canvases([contour1, contour2])
            merged_image = numpy.logical_or(*images).astype(numpy.uint8) * 255
            contours, _ = cv2.findContours(merged_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            return contours
        else:
            return None

    def _try_merge_contours(self, merged_contours: Dict[Tuple, numpy.ndarray]) -> bool:
        """
        Merges the first pair of overlapping contours inplace. If a successful merge was conducted, returns True.
        If all contours are non-overlapping the method returns False.
        """
        for first_box_key, second_box_key in combinations(merged_contours.keys(), 2):
            resulting_contour = self.merge_two_contours_if_overlapping(merged_contours[first_box_key],
                                                                       merged_contours[second_box_key])
            if resulting_contour is not None:
                merged_contours[first_box_key + second_box_key] = resulting_contour[0]
                merged_contours.pop(first_box_key)
                merged_contours.pop(second_box_key)
                return True
        return False

    def merge_contours(self, contours: List[numpy.ndarray], only_keep_overlapping: bool = False) -> List[numpy.ndarray]:
        """
        Iteratively try to find two contours that are overlapping. If an overlap is found, these contours will be
        merged. If there is no overlap anymore, merging is complete.
        """
        merged_contours = {(i,): contours[i] for i in range(len(contours))}
        new_merges = True
        while new_merges:
            new_merges = self._try_merge_contours(merged_contours)

        if only_keep_overlapping:
            overlapping_contours = [contour for ids, contour in merged_contours.items() if len(ids) > 1]
            return overlapping_contours
        else:
            return list(merged_contours.values())

    def merge_contours_of_same_class_from_different_images(self,
                                                           class_contours_for_sub_images: ClassContoursForSubImages,
                                                           batch_size: int,
                                                           only_keep_overlapping: bool = False,
                                                           class_names_to_merge: Tuple[str] = (),
                                                           drop_if_size_of_contours_zero: bool = False) \
            -> ClassContours:
        if len(class_names_to_merge) == 0:
            # we are merging all class ids we currently have in our class boxes
            class_names_to_merge = {class_name for sub_image_data in class_contours_for_sub_images.values()
                                    for class_name in sub_image_data.keys()}

        # (Indirectly) maps the class names to bboxes. Each dict value is a list of m items, where m is the number of
        # sub images. These lists consists of a lists with n items, where n is the number of batches. The contours are
        # represented by a list of ndarrays the represents the points of a polygon. ndarray is basically a list of
        # coordinates (x, y).
        # class_name: [
        #       sub image 1 = [
        #           batch 1 = [
        #               contour: ndarray(num_points, 1, 2)
        #           ]
        #           ...,
        #           batch n
        #       ]
        #       ...,
        #       sub image m
        # ]
        class_to_contours_dict = defaultdict(list)
        for class_contours in class_contours_for_sub_images.values():
            for class_name, contours in class_contours.items():
                class_to_contours_dict[class_name].append(contours)

        class_to_merged_contours_dict = defaultdict(list)
        for class_name, contours_for_class in class_to_contours_dict.items():
            for batch_id in range(batch_size):
                contours_for_current_batch = [contour[batch_id] for contour in contours_for_class]

                size_of_contours_is_zero = [len(contours) == 0 for contours in contours_for_current_batch]
                if all(size_of_contours_is_zero):
                    # there is nothing in this image to merge
                    class_to_merged_contours_dict[class_name].append(None)
                    continue
                if drop_if_size_of_contours_zero and class_name in class_names_to_merge and any(size_of_contours_is_zero):
                    # one image does not contain anything, we are dropping!
                    class_to_merged_contours_dict[class_name].append(None)
                    continue
                if any(size_of_contours_is_zero):
                    # no need to calculate IOUs we just keep the boxes with elements
                    for i, size_is_zero in enumerate(size_of_contours_is_zero):
                        if not size_is_zero:
                            class_to_merged_contours_dict[class_name].append(contours_for_class[i][batch_id])
                            break
                    continue
                num_sub_images = len(contours_for_current_batch)
                flat_contours = [contours for sub_image_contours in contours_for_current_batch for contours in sub_image_contours]
                if class_name not in class_names_to_merge or num_sub_images == 1:
                    # nothing to merge here, just keep the boxes
                    class_to_merged_contours_dict[class_name].append(flat_contours)
                    continue

                assert num_sub_images >= 2, "The dict item 'keys_for_class_determination' in the config file must " \
                                            "contain at least one item."
                merged_contours = self.merge_contours(flat_contours, only_keep_overlapping)

                if len(merged_contours) == 0:
                    class_to_merged_contours_dict[class_name].append(None)
                else:
                    class_to_merged_contours_dict[class_name].append(merged_contours)

        # returns a dict of contours for each class with shape {class_id: [[ndarray(num_points, 1, 2), ...], ...]}
        return class_to_merged_contours_dict

    def merge_contours_of_same_class_from_same_image(self, class_contours: ClassContours) -> ClassContours:
        all_merged_contours = {}
        for class_name, batch_contours_for_class in class_contours.items():
            merged_contours = []
            for contours_for_class in batch_contours_for_class:
                if contours_for_class is None:
                    merged_contours.append(None)
                    continue
                merged = self.merge_contours(contours_for_class)
                merged_contours.append(merged)
            all_merged_contours[class_name] = merged_contours
        return all_merged_contours

    def extract_contours(self, predicted_clusters: PredictedClusters, image_ids_to_extract: List[str]) \
            -> ClassContoursForSubImages:
        class_contours_for_sub_images = {}
        for key_id in image_ids_to_extract:
            cluster_class_tensors = predicted_clusters[key_id]
            tensors_for_class = {}
            for class_name, class_tensor in cluster_class_tensors.items():
                if class_name == "background":
                    continue
                # expected shape: [batch_size, height, width]
                cluster_arrays = class_tensor.cpu().numpy().astype(numpy.uint8)
                class_contours = self.cluster_image_to_contours(cluster_arrays)
                tensors_for_class[class_name] = class_contours
            class_contours_for_sub_images[key_id] = tensors_for_class
        return class_contours_for_sub_images

    def merge_finegrained_segmentation(self, predicted_clusters: PredictedClusters, batch_size: int) -> ClassContours:
        contours_for_sub_images = self.extract_contours(predicted_clusters, self.keys_for_finegrained_segmentation)

        # for each image in our batch, we only keep those images where contours of both steps overlap if one sub image
        # does not contain any contours, but the other does, we do not keep these images -> they do not contain anything
        # useful
        merged_contours = self.merge_contours_of_same_class_from_different_images(
            contours_for_sub_images,
            batch_size,
            only_keep_overlapping=True,
            drop_if_size_of_contours_zero=True,
        )
        if self.debug:
            self.render_debug_contours(merged_contours, "after_finegrained_merging")

        return merged_contours

    def classify_fine_grained_contours(self, text_regions_per_class: ClassContours,
                                       fine_grained_contours_per_class: ClassContours,
                                       fine_grained_class_name: str = 'printed_text') -> ClassContours:
        assert len(text_regions_per_class) == len(fine_grained_contours_per_class), \
            "Num classes of text regions and fine grained contours must be equal! "
        classified_contours = defaultdict(list)
        # we always use fine-grained contours of one class (default is printed text)
        fine_grained_contours_batches = fine_grained_contours_per_class[fine_grained_class_name]
        # sort the keys of text regions based on the order in color json file
        text_regions_per_class = dict(sorted(text_regions_per_class.items(), key=lambda x: self.class_id_map[x[0]]))

        # For each contour the size of the overlaps regarding each class is stored. In the end the contour is
        # classified as the class to which it had the largest overlap
        batch_size = len(fine_grained_contours_batches)
        class_ranking_for_contours = {i: defaultdict(dict) for i in range(batch_size)}
        for class_name, text_regions_batch in text_regions_per_class.items():
            for batch_id, (text_regions, fine_grained_contours) in enumerate(zip(text_regions_batch,
                                                                                 fine_grained_contours_batches)):
                all_text_regions_empty = text_regions is None
                fine_grained_contours_empty = fine_grained_contours is None or len(fine_grained_contours) == 0
                if all_text_regions_empty or fine_grained_contours_empty:
                    # there is nothing in this image
                    classified_contours[class_name].append(None)
                    continue

                for contour_id, fine_grained_contour in enumerate(fine_grained_contours):
                    class_ranking_for_current_contour = class_ranking_for_contours[batch_id][contour_id]
                    if class_name not in class_ranking_for_current_contour:
                        class_ranking_for_current_contour[class_name] = 0
                    for text_region in text_regions:
                        overlap = self.contour_overlap(fine_grained_contour, text_region)
                        class_ranking_for_current_contour[class_name] += overlap

        for class_name in text_regions_per_class.keys():
            # Initialization of batches: adds empty contour list for each
            classified_contours[class_name] = [[] for _ in range(batch_size)]
        for batch_id in range(batch_size):
            for contour_id, class_ranking in class_ranking_for_contours[batch_id].items():
                class_with_highest_overlap = max(class_ranking, key=class_ranking.get)
                if class_ranking[class_with_highest_overlap] > 0:
                    classified_contours[class_with_highest_overlap][batch_id].append(
                        fine_grained_contours_batches[batch_id][contour_id]
                    )
            for class_name in text_regions_per_class.keys():
                if len(classified_contours[class_name][batch_id]) == 0:
                    # No contours were added to this class, so we're setting the respective batch to None
                    classified_contours[class_name][batch_id] = None

        if self.debug:
            self.render_debug_contours(classified_contours, "classified_contours")
        return classified_contours

    def drop_too_small_contours(self, class_contours: ClassContours) -> ClassContours:
        adjusted_contours = {}
        for class_name, batch_contours in class_contours.items():
            adjusted_batch_contours = []
            for batch_id in range(len(batch_contours)):
                contours = batch_contours[batch_id]
                if contours is not None:
                    contours = [contour for contour in contours if cv2.contourArea(contour) >= self.min_class_contour_area]
                    if len(contours) == 0:
                        contours = None
                adjusted_batch_contours.append(contours)
            adjusted_contours[class_name] = adjusted_batch_contours
        return adjusted_contours

    def render_segmentation_image(self, fine_grained_prediction: Dict[str, torch.Tensor],
                                  classified_contours: ClassContours, batch_size: int,
                                  cluster_class_name: str = 'printed_text') -> numpy.ndarray:
        segmentation_images = []
        fine_grained_prediction = {class_name: clusters.cpu().numpy() for class_name, clusters in
                                   fine_grained_prediction.items()}
        for batch_id in range(batch_size):
            segmentation_image = numpy.zeros((self.image_size, self.image_size, 3), dtype=numpy.uint8)
            segmentation_image[:, :] = self.class_to_color_map['background']

            for class_name in fine_grained_prediction.keys():
                if class_name == 'background':
                    continue

                contours_for_class = classified_contours[class_name]
                image_contours = contours_for_class[batch_id]

                if image_contours is None:
                    # there is nothing in this image
                    continue

                for contour in image_contours:
                    # contours mask selects relevant region from fine_grained_prediction, which is used to colour the
                    # segmentation_image
                    contour_mask = numpy.zeros((self.image_size, self.image_size))
                    contour_mask = cv2.drawContours(contour_mask, [contour], 0, 1, cv2.FILLED)
                    contour_mask = contour_mask.astype(dtype=bool)
                    fine_grained_mask = numpy.where(
                        contour_mask,
                        fine_grained_prediction[cluster_class_name][batch_id],
                        False
                    )
                    segmentation_image[fine_grained_mask] = self.class_to_color_map[class_name]

            segmentation_images.append(segmentation_image)

        segmentation_images = numpy.stack(segmentation_images, axis=0)
        if self.debug:
            batch_size, height, width, num_channels = segmentation_images.shape
            extra = numpy.full((batch_size, self.max_debug_text_size, width, num_channels), 255, dtype='uint8')
            rgb_images = numpy.concatenate([segmentation_images, extra], axis=1)
            self.debug_images['result'] = rgb_images
        return segmentation_images

    def create_segmentation_image(self, activations: Dict[int, torch.Tensor]) -> Tuple[numpy.ndarray, List[int]]:
        raise NotImplementedError
