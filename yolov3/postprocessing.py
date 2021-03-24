import tensorflow as tf
from yolov3.configs import *

def _convert_boxes_from_xywh(boxes):
    # Split boxes and convert them to xmin, ymin, xmax, ymax
    boxes_x, boxes_y, boxes_w, boxes_h = tf.split(boxes, (1, 1, 1, 1), axis=-1)
    boxes_x1 = tf.clip_by_value(boxes_x - boxes_w * 0.5, clip_value_min=0, clip_value_max=1)
    boxes_y1 = tf.clip_by_value( boxes_y - boxes_h * 0.5, clip_value_min=0, clip_value_max=1)
    boxes_x2 = tf.clip_by_value(boxes_x + boxes_w * 0.5, clip_value_min=0, clip_value_max=1)
    boxes_y2 = tf.clip_by_value(boxes_y + boxes_h * 0.5, clip_value_min=0, clip_value_max=1)

    # Concatenate the transformed boxes
    boxes = tf.concat([boxes_x1, boxes_y1, boxes_x2, boxes_y2], axis=-1)

    return boxes

def _normalize_boxes(boxes):
    # Normalize xywh
    boxes /= YOLO_INPUT_SIZE
    boxes = tf.clip_by_value(boxes, clip_value_min=0, clip_value_max=1)

    return boxes

def _rearrange_pred_boxes(pred_boxes):
    # Flatten the boxes
    flattened_boxes = tf.reshape(pred_boxes, (-1, tf.shape(pred_boxes)[-1]))

    # Get xywh, conf, class_probs of each box
    boxes, box_conf, box_class_prob = \
        tf.split(flattened_boxes, (4, 1, -1), axis=-1)
    
    return boxes, box_conf, box_class_prob

def _get_box_scores(box_conf, box_class_prob):
    # Get box scores        
    box_scores = box_conf * tf.expand_dims(tf.reduce_max(box_class_prob, axis=-1), axis=-1)

    return box_scores

def _get_box_classes(box_class_prob):
    # Get box classes
    box_classes = tf.argmax(box_class_prob, axis=-1, output_type=tf.int32)
    box_classes = tf.expand_dims(box_classes, axis=-1)

    return box_classes

def _perform_nms(boxes, box_scores, box_classes, iou_threshold, score_threshold):
    # Apply non-max-suppression on the filtered boxes
    selected_idx, selected_box_scores = tf.image.non_max_suppression_with_scores(boxes, 
                                                tf.squeeze(box_scores, axis=-1),
                                                10, 
                                                iou_threshold=iou_threshold, 
                                                score_threshold=score_threshold)
    return selected_idx, selected_box_scores

def _filter_boxes(boxes, box_classes, selected_idx, selected_box_scores):
    
    # Filter selected_idx and box scores as output may be padded
    pad_filter_mask = selected_box_scores > 1e-6
    selected_box_scores = tf.expand_dims(selected_box_scores, axis=-1)

    selected_idx = selected_idx[pad_filter_mask]
    selected_box_scores = selected_box_scores[pad_filter_mask]

    # Filter boxes
    boxes = tf.gather(boxes, selected_idx)
    scores = selected_box_scores
    classes = tf.gather(box_classes, selected_idx)
    classes = tf.cast(classes, dtype=tf.float32) # must cast otherwise cannot concat

    return tf.concat((boxes, scores, classes), axis=-1)

def postprocess(pred_boxes, iou_threshold, score_threshold, filter_boxes=False, scale_up=False, original_image_size=None):
    boxes, box_conf, box_class_prob = _rearrange_pred_boxes(pred_boxes)

    boxes = _normalize_boxes(boxes)
    boxes = _convert_boxes_from_xywh(boxes)

    box_scores = _get_box_scores(box_conf, box_class_prob)
    box_classes = _get_box_classes(box_class_prob)

    selected_idx, selected_box_scores = _perform_nms(boxes, box_scores, box_classes, iou_threshold, score_threshold)

    if not filter_boxes:
        return boxes, box_scores, box_classes, selected_idx, selected_box_scores

    if scale_up and original_image_size != None:
        org_h, org_w = original_image_size
        boxes_x1, boxes_y1, boxes_x2, boxes_y2 = tf.split(boxes, (1, 1, 1, 1), axis=-1)
        boxes_x1 *= org_w
        boxes_x2 *= org_w
        boxes_y1 *= org_h
        boxes_y2 *= org_h

        boxes = tf.concat((boxes_x1, boxes_y1, boxes_x2, boxes_y2), axis=-1)
    
    combined_filtered_boxes = _filter_boxes(boxes, box_classes, selected_idx, selected_box_scores)

    return combined_filtered_boxes