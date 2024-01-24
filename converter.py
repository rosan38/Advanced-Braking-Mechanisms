from __future__ import print_function, division

import json
import os
import struct
from collections import defaultdict

from scipy.io import loadmat


def read_header(ifile):
    feed = ifile.read(4)
    norpix = ifile.read(24)
    version = struct.unpack('@i', ifile.read(4))
    length = struct.unpack('@i', ifile.read(4))
    assert length != 1024
    descr = ifile.read(512)
    params = [struct.unpack('@i', ifile.read(4))[0] for i in range(9)]
    fps = struct.unpack('@d', ifile.read(8))
    ifile.read(432)
    image_ext = {100: 'raw', 102: 'jpg', 201: 'jpg', 1: 'png', 2: 'png'}
    return {
        'w': params[0],
        'h': params[1],
        'bdepth': params[2],
        'ext': image_ext[params[5]],
        'format': params[5],
        'size': params[4],
        'true_size': params[8],
        'num_frames': params[6],
    }


def read_seq(path):
    assert path[-3:] == 'seq', path
    ifile = open(path, 'rb')
    params = read_header(ifile)
    bytes = open(path, 'rb').read()

    imgs = []
    extra = 8
    s = 1024
    for i in range(params['num_frames']):
        tmp = struct.unpack_from('@I', bytes[s:s + 4])[0]
        img = bytes[s + 4:s + tmp]
        s += tmp + extra
        if i == 0:
            val = struct.unpack_from('@B', bytes[s:s + 1])[0]
            if val != 0:
                s -= 4
            else:
                extra += 8
                s += 8
        imgs.append(img)

    return imgs, params['ext']


def read_vbb(path):
    assert path[-3:] == 'vbb'

    vbb = loadmat(path)
    nFrame = int(vbb['A'][0][0][0][0][0])
    objLists = vbb['A'][0][0][1][0]
    maxObj = int(vbb['A'][0][0][2][0][0])
    objInit = vbb['A'][0][0][3][0]
    objLbl = [str(v[0]) for v in vbb['A'][0][0][4][0]]
    objStr = vbb['A'][0][0][5][0]
    objEnd = vbb['A'][0][0][6][0]
    objHide = vbb['A'][0][0][7][0]
    altered = int(vbb['A'][0][0][8][0][0])
    log = vbb['A'][0][0][9][0]
    logLen = int(vbb['A'][0][0][10][0][0])

    data = {}
    data['nFrame'] = nFrame
    data['maxObj'] = maxObj
    data['log'] = log.tolist()
    data['logLen'] = logLen
    data['altered'] = altered
    data['frames'] = defaultdict(list)

    for frame_id, obj in enumerate(objLists):
        if obj.shape[1] > 0:
            for id, pos, occl, lock, posv in zip(obj['id'][0],
                                                 obj['pos'][0],
                                                 obj['occl'][0],
                                                 obj['lock'][0],
                                                 obj['posv'][0]):
                keys = obj.dtype.names
                id = int(id[0][0]) - 1
                p = pos[0].tolist()
                pos = [p[0] - 1, p[1] - 1, p[2], p[3]]
                occl = int(occl[0][0])
                lock = int(lock[0][0])
                posv = posv[0].tolist()

                datum = dict(zip(keys, [id, pos, occl, lock, posv]))
                datum['lbl'] = str(objLbl[datum['id']])
                datum['str'] = int(objStr[datum['id']]) - 1
                datum['end'] = int(objEnd[datum['id']]) - 1
                datum['hide'] = int(objHide[datum['id']])
                datum['init'] = int(objInit[datum['id']])

                data['frames'][frame_id].append(datum)

    return data


def extract_images_video(data_path, save_path, set_name):
    imgs, ext = read_seq(data_path)

    for idx, img in enumerate(imgs):
        img_fname = "{}_I{}.{}".format(set_name, str(idx).zfill(5), ext)
        img_path = os.path.join(save_path, img_fname)
        with open(img_path, 'wb+') as f:
            f.write(img)


def extract_annotations_video(data_path, save_path, set_name):
    data = read_vbb(data_path)

    for i in range(0, data['nFrame']):
        anno_fname = "{}_I{}.json".format(set_name, str(i).zfill(5))
        anno_path = os.path.join(save_path, anno_fname)
        try:
            with open(anno_path, 'w') as file_cache:
                json.dump(data['frames'][i],
                          file_cache,
                          sort_keys=True,
                          indent=4,
                          ensure_ascii=False)
        except IOError:
            raise IOError('Unable to open file: {}'.format(anno_path))


def extract_files(data_path, save_path, sets):
    sets = sets or ['set00', 'set01', 'set02', 'set03', 'set04', 'set05', 'set06', 'set07', 'set08', 'set09', 'set10']

    for j, set_name in enumerate(sets):
        set_path = os.path.join(data_path, set_name)
        set_path_annot = os.path.join(data_path, 'annotations', set_name)
        set_save_path = os.path.join(save_path, set_name)
        if not os.path.exists(set_save_path):
            os.makedirs(set_save_path)
        assert os.path.exists(set_path), 'File does not exists: {}'.format(set_path)
        print('\n> Extracting images + annotations from set: {} ({}/{})'.format(set_name, j + 1, len(sets)))
        fnames = os.listdir(set_path)
        fnames = [fname for fname in fnames if fname.endswith('.seq')]
        fnames.sort()
        for i, video in enumerate(fnames):
            video_name = os.path.splitext(video)[0]
            video_path = os.path.join(set_path, video_name + '.seq')
            annot_path = os.path.join(set_path_annot, video_name + '.vbb')
            img_save_path = os.path.join(set_save_path, 'images')
            annot_save_path = os.path.join(set_save_path, 'annotations')
            if not os.path.exists(img_save_path):
                os.makedirs(img_save_path)
            if not os.path.exists(annot_save_path):
                os.makedirs(annot_save_path)

            extract_images_video(video_path, img_save_path, set_name)

            extract_annotations_video(annot_path, annot_save_path, set_name)


def extract_data(data_path, save_path, sets=None):
    assert os.path.exists(data_path), "Must provide a valid data path: {}".format(data_path)
    assert save_path, "Must provide a valid storage path: {}".format(save_path)
    if sets:
        if isinstance(sets, str):
            sets = [sets]
        elif isinstance(sets, tuple) or isinstance(sets, list):
            sets = list(sets)
        else:
            raise TypeError('Invalid input type for \'sets\': {}.'.format(type(sets)))

    if not os.path.exists(save_path):
        print('> Saving extracted data to: {}'.format(save_path))
        os.makedirs(save_path)

    extract_files(data_path, save_path, sets)


if __name__ == '__main__':
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    data_path = ROOT_PATH + '/data_and_labels'
    save_path = ROOT_PATH + '/dataset'
    extract_data(data_path, save_path)
