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

### CNN output

Sigmoid for multi-label classification (last layer)
