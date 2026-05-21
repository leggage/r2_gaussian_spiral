import argparse
import csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary_csv', required=True)
    parser.add_argument('--output_txt', required=True)
    args = parser.parse_args()

    with open(args.summary_csv, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    lines = []
    for r in rows:
        case = r['case']
        lines.append(f"{case}_r2gaussian:")
        lines.append(f"psnr_2d: {r.get('psnr_2d_test','')}")
        lines.append(f"ssim_2d: {r.get('ssim_2d_test','')}")
        lines.append(f"psnr_3d: {r.get('psnr_3d','')}")
        lines.append(f"ssim_3d: {r.get('ssim_3d','')}")
        lines.append(f"time: {r.get('training_time_sec','')}")
        lines.append('')

    with open(args.output_txt, 'w', encoding='utf-8') as f:
        f.write('
'.join(lines).rstrip() + '
')

    print(f"Saved {args.output_txt}")


if __name__ == '__main__':
    main()
