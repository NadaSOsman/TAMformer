import os
import sys
import yaml
import numpy as np
import pickle
import cv2
import random as rn
import copy
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications import vgg16, resnet50, mobilenet
from tensorflow.keras.preprocessing.image import load_img
from pprint import pprint

class DataGenerator(Sequence):

    def __init__(self,
                 data=None,
                 labels=None,
                 data_sizes=None,
                 process=True,
                 global_pooling='max',
                 input_type_list=None,
                 batch_size=32,
                 shuffle=True,
                 to_fit=True,
                 stack_feats=False,
                 opts=None):

        self.data = data
        self.labels = labels
        self.process = process
        self.global_pooling = global_pooling
        self.input_type_list = input_type_list
        self.batch_size = 1 if len(self.labels[0]) < batch_size  else batch_size
        self.data_sizes = data_sizes
        self.shuffle = shuffle
        self.to_fit = to_fit
        self.stack_feats = stack_feats
        self.indices = None
        self.on_epoch_end()
        self.opts = opts

    def get_size(self):
        return len(self.data[0])

    def __len__(self):
        return int(np.floor(len(self.data[0])/self.batch_size))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.data[0]))
        self.pos_indices = []
        self.neg_indices = []
        for i in range(len(self.data[0])):
            if self.labels[0][i][0][0] == 0.0:
                self.neg_indices.append(i)
            else:
                self.pos_indices.append(i)
        self.neg_indices = np.array(self.neg_indices)
        self.pos_indices = np.array(self.pos_indices)
        if self.shuffle:
            np.random.shuffle(self.indices)
            np.random.shuffle(self.pos_indices)
            np.random.shuffle(self.neg_indices)

    def __getitem__(self, index):
        if self.batch_size == 1 or self.batch_size == 612:
            indices = self.indices[index*self.batch_size: (index+1)*self.batch_size]
        else:
            b_size = int(self.batch_size/2)
            indices = np.concatenate((self.pos_indices[min(index*b_size, len(self.pos_indices)-b_size): min((index+1)*b_size, len(self.pos_indices))],
                                      self.neg_indices[min(index*b_size, len(self.neg_indices)-b_size): min((index+1)*b_size, len(self.neg_indices))]), axis=-1)

        X = self._generate_X(indices)
        y = self._generate_y(indices)
        return X, y

    def _get_img_features(self, cached_path):
        with open(cached_path, 'rb') as fid:
            try:
                img_features = pickle.load(fid)
            except:
                img_features = pickle.load(fid, encoding='bytes')
        if self.process:
            if self.global_pooling == 'max':
                img_features = np.squeeze(img_features)
                img_features = np.amax(img_features, axis=0)
                img_features = np.amax(img_features, axis=0)
            elif self.global_pooling == 'avg':
                img_features = np.squeeze(img_features)
                img_features = np.average(img_features, axis=0)
                img_features = np.average(img_features, axis=0)
            else:
                img_features = img_features.ravel()
        return img_features

    def _normalize_box(self, X, label=False):
        new_X = X.copy()
        if label:
            new_X[:, 1] = new_X[:, 1]/1080.0
            new_X[:, 3] = new_X[:, 3]/1080.0
            new_X[:, 0] = new_X[:, 0]/1920.0
            new_X[:, 2] = new_X[:, 2]/1920.0
        else:
            new_X[:, :, 1] = new_X[:, :, 1]/1080.0
            new_X[:, :, 3] = new_X[:, :, 3]/1080.0
            new_X[:, :, 0] = new_X[:, :, 0]/1920.0
            new_X[:, :, 2] = new_X[:, :, 2]/1920.0
        return new_X

    def _normalize_VGG16(self, X):
        new_X = X.copy()/228.5
        return new_X

    def _normalize_speed(self, X):
        new_X = X.copy()/52.014
        return new_X

    def _generate_X(self, indices):
        X = []
        for input_type_idx, input_type in enumerate(self.input_type_list):
            features_batch = np.empty((self.batch_size, *self.data_sizes[input_type_idx]))
            for i, index in enumerate(indices):
                noise = np.random.normal(0,1, (self.data_sizes[input_type_idx][0], self.data_sizes[input_type_idx][1]))
                prob = rn.uniform(0,1)
                if isinstance(self.data[input_type_idx][index][0], str):
                    cached_path_list = self.data[input_type_idx][index]
                    for j, cached_path in enumerate(cached_path_list):
                        img_features = self._get_img_features(cached_path)

                        if len(cached_path_list) == 1:
                            features_batch[i, ] = img_features
                        else:
                            features_batch[i, j, ] = img_features
                else:
                    features_batch[i, ] = self.data[input_type_idx][index]
            if 'box' in input_type:
                X.append(self._normalize_box(features_batch))
            elif 'context' in input_type:
                X.append(self._normalize_VGG16(features_batch))
            else:
                X.append(features_batch)
        return X

    def _generate_y(self, indices):
        Y = np.empty((self.batch_size,))
        for i, index in enumerate(indices):
                Y[i, ] = self.labels[0][index][0][0]
        return [np.copy(Y) for i in range(int((self.opts['seq_len']-self.opts['obs_length'])/self.opts['step']))]




class DataGetter(object):

    def __init__(self, data_type, data_raw, model_opts):
        self.data_type = data_type
        self.data_raw = data_raw
        self.model_opts = model_opts
        self._generator = False
        self._global_pooling = 'max'
        self._backbone = 'vgg16'

    def get_data(self):
        self._generator = self.model_opts.get('generator', False)
        data_type_sizes_dict = {}
        process = self.model_opts.get('process', True)
        dataset = self.model_opts['dataset']
        data, intent_count, neg_count, pos_count = self.get_data_sequence()

        data_type_sizes_dict['box'] = data['box'].shape[1:]
        if 'speed' in data.keys():
            data_type_sizes_dict['speed'] = data['speed'].shape[1:]

        _data = []
        data_sizes = []
        data_types = []

        for d_type in self.model_opts['obs_input_type']:
            if 'local' in d_type or 'context' in d_type:
                features, feat_shape = self.get_context_data(data, d_type)
            elif 'pose' in d_type:
                path_to_pose = self.model_opts['path_to_pose']
                features = self.get_pose(data['image'],
                                    data['ped_id'],
                                    data_type=self.data_type,
                                    file_path=path_to_pose,
                                    dataset=self.model_opts['dataset'])
                feat_shape = features.shape[1:]
            else:
                features = data[d_type]
                feat_shape = features.shape[1:]

            _data.append(features)
            data_sizes.append(feat_shape)
            data_types.append(d_type)

        if self.data_type=='val':
            batch_size = self.model_opts['val_batch_size']
        elif self.data_type=='train':
            batch_size = self.model_opts['batch_size']

        if self._generator:
            _data = (DataGenerator(data=_data,
                                   labels=[data['crossing'], data['goals'], data['tte']],
                                   data_sizes=data_sizes,
                                   process=process,
                                   global_pooling=self._global_pooling,
                                   input_type_list=self.model_opts['obs_input_type'],
                                   batch_size=batch_size if (self.data_type=='train' or self.data_type=='val') else 1,
                                   shuffle=self.data_type != 'test',
                                   to_fit=self.data_type != 'test',
                                   opts=self.model_opts), data['labels'], data['lens'])
        else:
            _data = (_data, data['crossing'])

        return {'data': _data,
                'ped_id': data['ped_id'],
                'image': data['image'],
                'data_params': {'data_types': data_types, 'data_sizes': data_sizes},
                'count': {'neg_count': neg_count, 'pos_count': pos_count, 'intent_count':intent_count}}

    def get_data_sequence(self):
        d = {'center': self.data_raw['center'].copy(),
             'box': self.data_raw['bbox'].copy(),
             'ped_id': self.data_raw['pid'].copy(),
             'crossing': self.data_raw['activities'].copy(),
             'image': self.data_raw['image'].copy()}

        obs_length = self.model_opts['obs_length']
        time_to_event = self.model_opts['time_to_event']
        min_encoding_len = self.model_opts['min_encoding_len']

        try:
            d['speed'] = self.data_raw['obd_speed'].copy()
        except KeyError:
            d['speed'] = self.data_raw['vehicle_act'].copy()
            print('Jaad dataset does not have speed information')
            print('Vehicle actions are used instead')

        balance = self.model_opts['balance_data'] if self.data_type == 'train' else False
        if balance:
            self.balance_data_samples(d, self.data_raw['image_dimension'][0])

        d['lens'] = d['box'].copy()
        d['tte'] = d['box'].copy()
        d['goals'] = d['box'].copy()

        for k in d.keys():
            seqs = []
            lens = []
            for seq in d[k]:
                seq_len = len(seq)

                if len(seq)>136 and self.data_type == 'train' and self.model_opts['data_augmentation']:
                    seqs.extend([seq[i:i+16]+seq[len(seq)-4*time_to_event:] for i in range(0,len(seq)-2*obs_length-4*time_to_event,3)])
                    lens.extend([min(seq_len, obs_length+4*time_to_event) for i in range(0,len(seq)-2*obs_length-4*time_to_event,3)])

                if(int(len(seq)%(time_to_event+min_encoding_len))!=0):
                    seq = [seq[0]]*(time_to_event+min_encoding_len-int(len(seq)%(time_to_event+min_encoding_len)))+seq

                seqs.extend([seq[len(seq)-obs_length-4*time_to_event:]])
                lens.extend([min(seq_len, obs_length+4*time_to_event)])

            if k == 'lens':
                d[k] = lens
            else:
                d[k] = seqs

        for k in d.keys():
            d[k] = np.array(d[k])

        labels = []
        crossing_count = 0
        intending_count = 0
        for l in d['crossing']:
            labels.append(l[0][0])
            if(l[0][0] == 1):
                crossing_count +=1
            if(l[0][0] == 2):
                intending_count +=1
        d['labels'] = [[labels for i in range(int((self.model_opts['seq_len']-self.model_opts['obs_length'])/self.model_opts['step']))]]

        #del self.data_raw
        notcrossing_count = len(d['crossing']) - crossing_count - intending_count
        return d, intending_count, notcrossing_count, crossing_count

    def update_progress(self, progress):
        barLength = 20  # Modify this to change the length of the progress bar
        status = ""
        if isinstance(progress, int):
            progress = float(progress)

        block = int(round(barLength * progress))
        text = "\r[{}] {:0.2f}% {}".format("#" * block + "-" * (barLength - block), progress * 100, status)
        sys.stdout.write(text)
        sys.stdout.flush()

    def balance_data_samples(self, d, img_width, balance_tag='crossing'):
        print("Balancing with respect to {} tag".format(balance_tag))
        gt_labels = [gt[0] for gt in d[balance_tag]]
        num_pos_samples = np.count_nonzero(np.array(gt_labels))
        num_neg_samples = len(gt_labels) - num_pos_samples

        # finds the indices of the samples with larger quantity
        if num_neg_samples == num_pos_samples:
            print('Positive and negative samples are already balanced')
        else:
            print('Unbalanced: \t Positive: {} \t Negative: {}'.format(num_pos_samples, num_neg_samples))
            if num_neg_samples > num_pos_samples:
                gt_augment = 1
            else:
                gt_augment = 0

            num_samples = len(d[balance_tag])
            for i in range(num_samples):
                if d[balance_tag][i][0][0] == gt_augment:
                    for k in d:
                        if k == 'center':
                            flipped = d[k][i].copy()
                            flipped = [[img_width - c[0], c[1]]
                                       for c in flipped]
                            d[k].append(flipped)
                        if k == 'box':
                            flipped = d[k][i].copy()
                            flipped = [np.array([img_width - b[2], b[1], img_width - b[0], b[3]])
                                       for b in flipped]
                            d[k].append(flipped)
                        if k == 'image':
                            flipped = d[k][i].copy()
                            flipped = [im.replace('.png', '_flip.png') for im in flipped]
                            d[k].append(flipped)
                        if k in ['speed', 'ped_id', 'crossing', 'walking', 'looking']:
                            d[k].append(d[k][i].copy())

            gt_labels = [gt[0] for gt in d[balance_tag]]
            num_pos_samples = np.count_nonzero(np.array(gt_labels))
            num_neg_samples = len(gt_labels) - num_pos_samples
            if num_neg_samples > num_pos_samples:
                rm_index = np.where(np.array(gt_labels) == 0)[0]
            else:
                rm_index = np.where(np.array(gt_labels) == 1)[0]

            # Calculate the difference of sample counts
            dif_samples = abs(num_neg_samples - num_pos_samples)
            # shuffle the indices
            np.random.seed(42)
            np.random.shuffle(rm_index)
            # reduce the number of indices to the difference
            rm_index = rm_index[0:dif_samples]

            # update the data
            for k in d:
                seq_data_k = d[k]
                d[k] = [seq_data_k[i] for i in range(0, len(seq_data_k)) if i not in rm_index]

            new_gt_labels = [gt[0] for gt in d[balance_tag]]
            num_pos_samples = np.count_nonzero(np.array(new_gt_labels))
            print('Balanced:\t Positive: %d  \t Negative: %d\n'
                  % (num_pos_samples, len(d[balance_tag]) - num_pos_samples))

    def get_context_data(self, data, feature_type):
        process = self.model_opts.get('process', True)
        aux_name = [self._backbone]
        if not process:
            aux_name.append('raw')
        aux_name = '_'.join(aux_name).strip('_')
        eratio = self.model_opts['enlarge_ratio']
        dataset = self.model_opts['dataset']

        data_gen_params = {'data_type': self.data_type, 'crop_type': 'none',
                           'target_dim': self.model_opts.get('target_dim', (224, 224))}
        if 'local_box' in feature_type:
            data_gen_params['crop_type'] = 'bbox'
            data_gen_params['crop_mode'] = 'pad_resize'
        elif 'local_context' in feature_type:
            data_gen_params['crop_type'] = 'context'
            data_gen_params['crop_resize_ratio'] = eratio
        elif 'surround' in feature_type:
            data_gen_params['crop_type'] = 'surround'
            data_gen_params['crop_resize_ratio'] = eratio
        elif 'scene_context' in feature_type:
            data_gen_params['crop_type'] = 'none'
        save_folder_name = feature_type
        save_folder_name = '_'.join([feature_type, aux_name])
        if 'local_context' in feature_type or 'surround' in feature_type:
            save_folder_name = '_'.join([save_folder_name, str(eratio)])
        data_gen_params['save_path'],_ = self.get_path(save_folder=save_folder_name,
                                         dataset=dataset, save_root_folder='data/features')
        return self.load_images_crop_and_process(data['image'],
                                                 data['box'],
                                                 data['ped_id'],
                                                 process=process,
                                                 **data_gen_params)


    def jitter_bbox(self, img_path, bbox, mode, ratio):
        assert (mode in ['same', 'enlarge', 'move', 'random_enlarge', 'random_move']), \
        'mode %s is invalid.' % mode

        if mode == 'same':
            return bbox

        img = load_img(img_path)

        if mode in ['random_enlarge', 'enlarge']:
            jitter_ratio = abs(ratio)
        else:
            jitter_ratio = ratio

        if mode == 'random_enlarge':
            jitter_ratio = np.random.random_sample() * jitter_ratio
        elif mode == 'random_move':
            # for ratio between (-jitter_ratio, jitter_ratio)
            # for sampling the formula is [a,b), b > a,
            # random_sample * (b-a) + a
            jitter_ratio = np.random.random_sample() * jitter_ratio * 2 - jitter_ratio

        jit_boxes = []
        for b in bbox:
            bbox_width = b[2] - b[0]
            bbox_height = b[3] - b[1]

            width_change = bbox_width * jitter_ratio
            height_change = bbox_height * jitter_ratio

            if width_change < height_change:
                height_change = width_change
            else:
                width_change = height_change

            if mode in ['enlarge', 'random_enlarge']:
                b[0] = b[0] - width_change // 2
                b[1] = b[1] - height_change // 2
            else:
                b[0] = b[0] + width_change // 2
                b[1] = b[1] + height_change // 2

            b[2] = b[2] + width_change // 2
            b[3] = b[3] + height_change // 2

            # Checks to make sure the bbox is not exiting the image boundaries
            b = self.bbox_sanity_check(img.size, b)
            jit_boxes.append(b)
        # elif crop_opts['mode'] == 'border_only':
        return jit_boxes

    def bbox_sanity_check(self, img_size, bbox):
        img_width, img_heigth = img_size
        if bbox[0] < 0:
            bbox[0] = 0.0
        if bbox[1] < 0:
            bbox[1] = 0.0
        if bbox[2] >= img_width:
            bbox[2] = img_width - 1
        if bbox[3] >= img_heigth:
            bbox[3] = img_heigth - 1
        return bbox


    def img_pad(self, img, mode='warp', size=224):
        assert (mode in ['same', 'warp', 'pad_same', 'pad_resize', 'pad_fit']), 'Pad mode %s is invalid' % mode
        image = np.copy(img)
        if mode == 'warp':
            warped_image = cv2.resize(img, (size, size))
            return warped_image
        elif mode == 'same':
            return image
        elif mode in ['pad_same', 'pad_resize', 'pad_fit']:
            img_size = image.shape[:2][::-1] # original size is in (height, width)
            ratio = float(size)/max(img_size)
            if mode == 'pad_resize' or \
              (mode == 'pad_fit' and (img_size[0] > size or img_size[1] > size)):
                img_size = tuple([int(img_size[0] * ratio), int(img_size[1] * ratio)])
                image = cv2.resize(image, img_size)
            padded_image = np.zeros((size, size)+(image.shape[-1],), dtype=img.dtype)
            w_off = (size-img_size[0])//2
            h_off = (size-img_size[1])//2
            padded_image[h_off:h_off + img_size[1], w_off:w_off+ img_size[0],:] = image
            return padded_image


    def squarify(self, bbox, squarify_ratio, img_width):
        width = abs(bbox[0] - bbox[2])
        height = abs(bbox[1] - bbox[3])
        width_change = height * squarify_ratio - width
        bbox[0] = bbox[0] - width_change / 2
        bbox[2] = bbox[2] + width_change / 2
        # Squarify is applied to bounding boxes in Matlab coordinate starting from 1
        if bbox[0] < 0:
            bbox[0] = 0

        # check whether the new bounding box goes beyond image boarders
        # If this is the case, the bounding box is shifted back
        if bbox[2] > img_width:
            # bbox[1] = str(-float(bbox[3]) + img_dimensions[0])
            bbox[0] = bbox[0] - bbox[2] + img_width
            bbox[2] = img_width
        return bbox


    def load_images_crop_and_process(self, img_sequences, bbox_sequences,
                                     ped_ids, save_path,
                                     data_type='train',
                                     crop_type='none',
                                     crop_mode='warp',
                                     crop_resize_ratio=2,
                                     target_dim=(224, 224),
                                     process=True,
                                     regen_data=False):

        print("Generating {} features crop_type={} crop_mode={}\
              \nsave_path={}, ".format(data_type, crop_type, crop_mode,
                                       save_path))
        preprocess_dict = {'vgg16': vgg16.preprocess_input, 'resnet50': resnet50.preprocess_input, 'mobilenet': mobilenet.preprocess_input}
        backbone_dict = {'vgg16': vgg16.VGG16, 'resnet50': resnet50.ResNet50, 'mobilenet': mobilenet.MobileNet}
        print("Backbone Models Loaded ......")

        preprocess_input = preprocess_dict.get(self._backbone, None)
        print("Preprocessing Model:", self._backbone)
        if process:
            assert (self._backbone in ['vgg16', 'resnet50', 'mobilenet']), "{} is not supported".format(self._backbone)

        print("Initializing Preprocessin Model.......")
        convnet =  backbone_dict[self._backbone](input_shape=(224, 224, 3), include_top=False, weights="imagenet")

        print("Preprocessing Model Initialized........")
        sequences = []
        bbox_seq = bbox_sequences.copy()
        i = -1
        for seq, pid in zip(img_sequences, ped_ids):
            i += 1
            self.update_progress(i / len(img_sequences))
            img_seq = []
            for imp, b, p in zip(seq, bbox_seq[i], pid):
                flip_image = False
                set_id = imp.split('/')[-3]
                vid_id = imp.split('/')[-2]
                img_name = imp.split('/')[-1].split('.')[0]
                img_save_folder = os.path.join(save_path, set_id, vid_id)

                # Modify the path depending on crop mode
                if crop_type == 'none':
                    img_save_path = os.path.join(img_save_folder, img_name + '.pkl')
                else:
                    img_save_path = os.path.join(img_save_folder, img_name + '_' + p[0] + '.pkl')

                # Check whether the file exists
                if os.path.exists(img_save_path) and not regen_data:
                    if not self._generator:
                        with open(img_save_path, 'rb') as fid:
                            try:
                                img_features = pickle.load(fid)
                            except:
                                img_features = pickle.load(fid, encoding='bytes')
                else:
                    if 'flip' in imp:
                        imp = imp.replace('_flip', '')
                        flip_image = True
                    if crop_type == 'none':
                        img_data = cv2.imread(imp)
                        img_features = cv2.resize(img_data, target_dim)
                        if flip_image:
                            img_features = cv2.flip(img_features, 1)
                    else:
                        img_data = cv2.imread(imp)
                        if flip_image:
                            img_data = cv2.flip(img_data, 1)
                        if crop_type == 'bbox':
                            b = list(map(int, b[0:4]))
                            cropped_image = img_data[b[1]:b[3], b[0]:b[2], :]
                            img_features = self.img_pad(cropped_image, mode=crop_mode, size=target_dim[0])
                        elif 'context' in crop_type:
                            bbox = self.jitter_bbox(imp, [b], 'enlarge', crop_resize_ratio)[0]
                            bbox = self.squarify(bbox, 1, img_data.shape[1])
                            bbox = list(map(int, bbox[0:4]))
                            cropped_image = img_data[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                            img_features = self.img_pad(cropped_image, mode='pad_resize', size=target_dim[0])
                        elif 'surround' in crop_type:
                            b_org = list(map(int, b[0:4])).copy()
                            bbox = self.jitter_bbox(imp, [b], 'enlarge', crop_resize_ratio)[0]
                            bbox = self.squarify(bbox, 1, img_data.shape[1])
                            bbox = list(map(int, bbox[0:4]))
                            img_data[b_org[1]:b_org[3], b_org[0]:b_org[2], :] = 128
                            cropped_image = img_data[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                            img_features = self.img_pad(cropped_image, mode='pad_resize', size=target_dim[0])
                        else:
                            raise ValueError('ERROR: Undefined value for crop_type {}!'.format(crop_type))
                    if preprocess_input is not None:
                        img_features = preprocess_input(img_features)
                    if process:
                        expanded_img = np.expand_dims(img_features, axis=0)
                        img_features = convnet.predict(expanded_img, verbose=0)
                    # Save the file
                    if not os.path.exists(img_save_folder):
                        os.makedirs(img_save_folder)
                    with open(img_save_path, 'wb') as fid:
                        pickle.dump(img_features, fid, pickle.HIGHEST_PROTOCOL)

                # if using the generator save the cached features path and size of the features
                if process and not self._generator:
                    if self._global_pooling == 'max':
                        img_features = np.squeeze(img_features)
                        img_features = np.amax(img_features, axis=0)
                        img_features = np.amax(img_features, axis=0)
                    elif self._global_pooling == 'avg':
                        img_features = np.squeeze(img_features)
                        img_features = np.average(img_features, axis=0)
                        img_features = np.average(img_features, axis=0)
                    else:
                        img_features = img_features.ravel()

                if self._generator:
                    img_seq.append(img_save_path)
                else:
                    img_seq.append(img_features)
            sequences.append(img_seq)
        sequences = np.array(sequences)
        # compute size of the features after the processing
        if self._generator:
            with open(sequences[0][0], 'rb') as fid:
                feat_shape = pickle.load(fid).shape
            if process:
                if self._global_pooling in ['max', 'avg']:
                    feat_shape = feat_shape[-1]
                else:
                    feat_shape = np.prod(feat_shape)
            if not isinstance(feat_shape, tuple):
                feat_shape = (feat_shape,)
            feat_shape = (np.array(bbox_sequences).shape[1],) + feat_shape
        else:
            feat_shape = sequences.shape[1:]

        return sequences, feat_shape


    def get_path(self, file_name='',
             sub_folder='',
             save_folder='models',
             dataset='jaad',
             save_root_folder='data/'):

        save_path = os.path.join(save_root_folder, dataset, save_folder, sub_folder)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return os.path.join(save_path, file_name), save_path


    def flip_pose(self, pose):
        flip_map = [0, 1, 2, 3, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7, 8, 9, 22, 23, 24, 25,
                    26, 27, 16, 17, 18, 19, 20, 21, 30, 31, 28, 29, 34, 35, 32, 33]
        new_pose = pose.copy()
        flip_pose = [0] * len(new_pose)
        for i in range(len(new_pose)):
            if i % 2 == 0 and new_pose[i] != 0:
                new_pose[i] = 1 - new_pose[i]
            flip_pose[flip_map[i]] = new_pose[i]
        return flip_pose


    def get_pose(self, img_sequences, ped_ids, file_path, data_type='train', dataset='jaad'):
        poses_all = []
        set_poses_list = [x for x in os.listdir(file_path) if x.endswith('.pkl')]
        set_poses = {}
        for s in set_poses_list:
            with open(os.path.join(file_path, s), 'rb') as fid:
                try:
                    p = pickle.load(fid)
                except:
                    p = pickle.load(fid, encoding='bytes')
            set_poses[s.split('.pkl')[0].split('_')[-1]] = p
        i = -1
        for seq, pid in zip(img_sequences, ped_ids):
            i += 1
            #update_progress(i / len(img_sequences))
            pose = []
            for imp, p in zip(seq, pid):
                flip_image = False

                if dataset == 'pie':
                    set_id = imp.split('/')[-3]
                elif dataset == 'jaad':
                    set_id = 'set01'

                vid_id = imp.split('/')[-2]
                img_name = imp.split('/')[-1].split('.')[0]
                if 'flip' in img_name:
                    img_name = img_name.replace('_flip', '')
                    flip_image = True
                k = img_name + '_' + p[0]
                if(vid_id not in set_poses[set_id]):
                    if(len(pose) != 0):
                        pose.append(pose[-1])
                    else:
                        pose.append([0] * 36)
                elif k in set_poses[set_id][vid_id].keys():
                    # [nose, neck, Rsho, Relb, Rwri, Lsho, Lelb, Lwri, Rhip, Rkne,
                    #  Rank, Lhip, Lkne, Lank, Leye, Reye, Lear, Rear, pt19]
                    if flip_image:
                        pose.append(self.flip_pose(set_poses[set_id][vid_id][k]))
                    else:
                        pose.append(set_poses[set_id][vid_id][k])
                else:
                    pose.append([0] * 36)
            #print(pose)
            poses_all.append(pose)
        poses_all = np.array(poses_all)
        return poses_all
