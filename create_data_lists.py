import argparse
from utils import create_data_lists


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', required=True, choices=('clipart', 'watercolor', 'comic'))
    parser.add_argument('--train_type', required=True, choices=('ideal', 'dt', 'dt_pl', 'p_dt', 'p_dt_pl'))
    args = parser.parse_args()

    if args.train_type == 'ideal':
        voc07_path1 = f'./dataset/{args.data_type}'
        voc07_path2 = voc12_path1 = voc12_path2 = None
        output_folder = f'./json/{args.data_type}'
    elif args.train_type in ('dt', 'p_dt'):
        voc07_path1 = f'./dataset/{args.data_type}_dt/VOC2007'
        voc12_path1 = f'./dataset/{args.data_type}_dt/VOC2012'
        voc07_path2 = voc12_path2 = None
        output_folder = f'./json/{args.data_type}_dt'
    elif args.train_type in ('dt_pl', 'p_dt_pl'):
        voc07_path1 = f'./dataset/{args.data_type}_dt_pl'
        voc07_path2 = voc12_path1 = voc12_path2 = None
        output_folder = f'./json/{args.data_type}_dt_pl'
    if 'p' == args.train_type[0]:
        voc07_path2 = './dataset/pascal_voc/VOC2007'
        voc12_path2 = './dataset/pascal_voc/VOC2012'
        output_folder = output_folder[:7] + 'p_' + output_folder[7:]

    # print(voc07_path1)
    # print(voc07_path2)
    # print(voc12_path1)
    # print(voc12_path2)
    # print(output_folder)
    
    create_data_lists(voc07_path1=voc07_path1,
                      voc07_path2=voc07_path2,
                      voc12_path1=voc12_path1,
                      voc12_path2=voc12_path2,
                      output_folder=output_folder,
                      type=args.train_type)
