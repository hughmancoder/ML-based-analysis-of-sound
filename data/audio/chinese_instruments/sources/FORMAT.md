# FORMAT

```json
[
  {
    "file": "path/to/local/file.mp3",
    "video": "https://youtube.com/…",       // either `file` or `video` is required
    "labels": ["label1","label2"],                     // list of labels (primary = first element)
    "time_stamps_to_extract": [["0:18", "2:05"]],          // [] = full file, or list of [start, end] ranges in MM:SS format
    "description": "optional description"
  }
]