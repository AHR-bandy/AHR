# AHR
Application of Handwriting Recognition
===

1. 画像認識の基本的な考え方（白黒文字）
1. 全ての画像ファイルで縦横を分割する（準備過程）
1. 分割して生成されたブロック群の中で各セルに学習データで同傾向の多さを基準に重みを付ける（学習過程）
1. 重みを基に最小二乗法等によって検査する文字がどの文字にあたるかのもっともらしさを求め、最も値の高い文字を出力する（テスト過程）
