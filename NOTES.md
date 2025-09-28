# Structure (recommended)

## Structure suggestions

[template example](https://github.com/victoresque/pytorch-template)

## Dataset (IRMAS)

[Data set can be found here](https://zenodo.org/records/1290750#.WzCwSRyxXMU)
Specs: 44.1 kHz, 3.0 s 16 bit stereo WAV

## Focus

4 instruments: gong, erhu, dizi, pipa
- Extract audio from 3 target films
- Segment into ~30 ms windows 

## CNN output

Sigmoid for multi-label classification (last layer)

## Train Log

### Train train_irmas.py

Best val acc: 0.5632

## Checkpoint

[001/30] train 1.6473/0.4288 | val 1.4015/0.5610 | time 19.3s

