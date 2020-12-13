import os.path as osp
from datasets.roidb import Roidb
from datasets.refer import Refer
from opt import parse_opt
import json
import io

opt = parse_opt()
opt = vars(opt)


class Refvg(object):

    def __init__(self, split, model_method):
        self._dataset = 'refvg'
        self._imageset = 'vg'
        self._split = split
        # load the *_expression.json
        self._ref_db = Refer(opt['data_root'], self._dataset, split)
        # load the *_sgs.json and *_sg_seqs.json
        if model_method == 'sgmn':
            self._ref_sg = self._load_sg()
            self._ref_sg_seq = self._load_sg_seq()
        else:
            self._ref_sg = None
            self._ref_sg_seq = None
        # Get the ids of all the referent expressions
        self._sent_ids = self._ref_db.get_sentIds()
        self._image_ids = self._ref_db.get_imgIds(self._sent_ids)
        roidb = Roidb(self._imageset, model_method)
        # Get the corresponding image id and remove duplicate
        # self.sent_ids = self.ref_db.remove_redundant_sent(self._sent_ids, roidb.image_ids)
        self._rois_db = {}
        self.max_num_box = 0
        for img_id in self._image_ids:
            assert roidb.roidb.has_key(img_id)
            # roidb.roidb is the content of gt_objects_info.json
            self._rois_db[img_id] = roidb.roidb[img_id].copy()
            # the largest number of objects in a image
            self.max_num_box = max(self.max_num_box, int(self._rois_db[img_id]['num_objs']))
        self._h5_files = roidb.h5_files # the file descriptor of gt_object_%d.h5
        self._h5_lrel_files = roidb.h5_lrel_files

    @property
    def sent_ids(self):
        return self._sent_ids

    @property
    def ref_db(self):
        return self._ref_db

    @property
    def image_ids(self):
        return self._image_ids

    @property
    def rois_db(self):
        return self._rois_db

    @property
    def h5_files(self):
        return self._h5_files

    @property
    def h5_lrel_files(self):
        return self._h5_lrel_files

    @property
    def ref_sg(self):
        return self._ref_sg

    @property
    def ref_sg_seq(self):
        return self._ref_sg_seq

    @property
    def id_to_path(self):
        path = {}
        for img_id in self.image_ids:
            file_name = str(img_id)+ '.jpg'
            image_path = osp.join(opt['data_root'], 'images/') + file_name
            path[img_id] = image_path
        return path

    def get_imgIds(self, sent_ids):
        return self._ref_db.get_imgIds(sent_ids)

    def _load_sg(self):
        sgs = {}
        sg_file_path = osp.join(opt['data_root'], self._dataset, self._split + '_sgs.json')
        data = json.load(open(sg_file_path, 'r'))
        for key in list(data.keys()):
            sgs[key] = data[key]
        return sgs

    def _load_sg_seq(self):
        sg_seqs = {}
        sg_seq_file_path = osp.join(opt['data_root'], self._dataset, self._split + '_sg_seqs.json')
        data = json.load(open(sg_seq_file_path, 'r'))
        for key in list(data.keys()):
            sg_seqs[key] = data[key]
        return sg_seqs

    def load_dictionary(self, pad_at_first=True):
        dict_file = osp.join(opt['data_root'], 'word_embedding', 'vocabulary_72700.txt')
        with io.open(dict_file, encoding='utf-8') as f:
            words = [w.strip() for w in f.readlines()]
        if pad_at_first and words[0] != '<pad>':
            raise Exception("The first word needs to be <pad> in the word list.")
        vocab_dict = {words[n]: n for n in range(len(words))}
        return vocab_dict

    def get_img_path(self, id):
        return self.id_to_path[id]

    def get_sent(self, sent_id):
        return self.ref_db.load_sent(sent_id)