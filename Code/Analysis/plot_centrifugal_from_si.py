"""Plot the deposited centrifugal measurements and pressure-boundary table."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WORKBOOK = ROOT / 'Source_Data/Centrifugal liquid loss test/Centrifugal liquid loss test.xlsx'
PRESSURE = ROOT / 'Source_Data/Centrifugal_pressure_complete_series.csv'


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_observations(path):
    sheet = pd.read_excel(path, sheet_name=0, header=None)
    records = []
    for column, radius in [(0, .5), (2, .4), (4, .3)]:
        label = str(sheet.iloc[1, column]).lower().replace(' ', '')
        if label != f'{radius:g}mm':
            raise ValueError(f'Unexpected radius label in column {column + 1}: {label}')
        for row in range(2, len(sheet)):
            speed, loss = sheet.iloc[row, column:column + 2]
            if pd.isna(speed) and pd.isna(loss):
                continue
            if pd.isna(speed) or pd.isna(loss):
                raise ValueError(f'Incomplete measurement pair at worksheet row {row + 1}.')
            speed, loss = float(speed), float(loss)
            if not np.isfinite([speed, loss]).all() or speed < 0 or not 0 <= loss <= 100:
                raise ValueError(f'Invalid measurement at worksheet row {row + 1}.')
            records.append(dict(specimen_radius_mm=radius, speed_rpm=speed,
                                mass_loss_pct=loss, worksheet_row=row + 1))
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workbook', default=str(WORKBOOK))
    parser.add_argument('--pressure-table', default=str(PRESSURE))
    parser.add_argument('--out', default='ReviewChecks/figures')
    args = parser.parse_args()
    observations = load_observations(project_path(args.workbook))
    pressure = pd.read_csv(project_path(args.pressure_table)).iloc[:, :4].copy()
    pressure.columns = ['speed_rpm', 'centrifugal_pressure_pa', 'total_pressure_pa', 'critical_radius_mm']
    pressure = pressure.apply(pd.to_numeric, errors='raise').sort_values('speed_rpm')
    if not np.isfinite(pressure.to_numpy()).all():
        raise ValueError('The pressure table contains non-finite values.')
    output = project_path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    observations.to_csv(output / 'centrifugal_observations.csv', index=False)
    pressure.to_csv(output / 'centrifugal_pressure_boundary.csv', index=False)
    colors = {0.5: '#2171B5', 0.4: '#BCC0DF', 0.3: '#FFCFCF'}
    markers = {0.5: '*', 0.4: 'D', 0.3: 'P'}
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'svg.fonttype': 'none',
                         'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, (left, right) = plt.subplots(1, 2, figsize=(7, 2.8), layout='constrained')
    for radius, group in observations.groupby('specimen_radius_mm', sort=False):
        group = group.sort_values('speed_rpm', kind='stable')
        left.step(group.speed_rpm, group.mass_loss_pct, where='post', color=colors[radius], lw=1.2)
        left.scatter(group.speed_rpm, group.mass_loss_pct, s=35, color=colors[radius],
                     marker=markers[radius], edgecolor='black', linewidth=.4, label=f'{radius:.1f} mm')
        right.axhline(radius, color=colors[radius], lw=.7, ls='--')
    left.set(xlim=(0, 830), ylim=(-3, 70), xlabel='Rotation speed (rpm)',
             ylabel='Mass loss (%)', title='h  Centrifugal liquid loss')
    left.legend(frameon=False, title='Equivalent radius')
    right.plot(pressure.speed_rpm, pressure.critical_radius_mm, color='#183B56', lw=1.6,
               label='Calculated boundary')
    right.set(xlim=(0, 1000), ylim=(0, .62), xlabel='Rotation speed (rpm)',
              ylabel='Critical equivalent radius (mm)', title='i  Calculated stability boundary')
    right.legend(frameon=False)
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(output / f'Figure3_hi.{extension}', dpi=350, facecolor='white')
    plt.close(fig)
    print(f'Saved {len(observations)} workbook observations, the archived boundary and Figure3 h-i in {output}')


if __name__ == '__main__':
    main()
