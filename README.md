# gamut-probe

画像処理アプリのラウンドトリップテスト（画像を開いて何も編集せずに書き出し、元の画像と比べるテスト）で、色域とビット深度がそのまま保たれるかを測るツールです。

Evoto で TIFF/16bit を選んで書き出しても、中身が 8bit 精度になることを確認するために作りました。Evoto 以外のアプリのラウンドトリップテストにも使えます。

- `gamut_probe.py`：テストチャートの作成と、戻ってきた TIFF の解析
- `evoto_restore.py`：Evoto の出力のうち、レタッチされていない画素を元の 16bit データに戻す

## 必要なもの

[uv](https://docs.astral.sh/uv/) だけです。Python と依存ライブラリ（numpy、tifffile、imagecodecs、Pillow）は、初回の `uv run` で自動的にそろいます。

```bash
brew install uv
# または
curl -LsSf https://astral.sh/uv/install.sh | sh
```

ファイルはすべて同じフォルダに置いてください。`evoto_restore.py` は `gamut_probe.py` を読み込みます。

## 追試の手順

### 1. テストチャートを作る

```bash
uv run gamut_probe.py make -o charts
```

`charts/` に次のファイルができます。

- `probe_DisplayP3_16bit.tif`：実際のワークフロー（Display P3 で渡す）と同じ条件で測る用
- `probe_ProPhoto_16bit.tif`：アプリの色域の限界を探る用
- それぞれの `.layout.json`（パッチの位置情報）と ICC プロファイル

`.tif` と `.layout.json` は、同じフォルダに置いたままにしてください。

### 2. アプリに通す

1. チャートをアプリで開きます
2. 何も編集しません。トリミング、回転、リサイズもしません（Evoto の場合は、自動カラーと肌トーン統一も OFF）
3. TIFF/16bit・原寸で、シャープなしで書き出します

### 3. 解析する

```bash
uv run gamut_probe.py analyze charts/probe_DisplayP3_16bit.tif Evoto_P3.tif -o report.html
```

1 つ目の引数が元のチャート、2つ目以降がアプリから戻ってきたファイルです。複数のアプリの結果を並べることもできます。

```bash
uv run gamut_probe.py analyze charts/probe_DisplayP3_16bit.tif Evoto_P3.tif Luminar_P3.tif -o report.html
```

ターミナルに判定の要約が出て、詳細は HTML レポートに書き出されます。

## 判定の見方

- 色域：グレー、sRGB の内側、sRGB の外側、Adobe RGB だけに入る色、P3 だけに入る色、どちらにも入らない色（ProPhoto 版のみ）のパッチごとに、元の色との差（ΔE00）を見ます。目安は、1 未満でほぼ見分けがつかない、2 未満で保持、5 以上ではっきり変化です
- ビット深度：0.45〜0.47 の微小グラデーションに、何段の階調が残っているかを見ます。元のチャートは 1,311 段で、8bit 精度に落ちていると数段しか残りません

### 作者の環境での結果（macOS、Evoto 7 / 8）

- Evoto 7 / 8：Adobe RGB として書き出され、その外側の色はクリップ。微小グラデーションは 6 段で、全画素が 8bit 精度
- Luminar（ProPhoto/16bit で書き出し）：色域は保持。微小グラデーションは 1,281 段
- Affinity：色域は保持。微小グラデーションは 1,311 段

Evoto 7 と 8 の出力は、画素単位で完全に一致しました。Windows 版や、RAW を直接読み込んだ場合は未確認です。結果を issue で教えてもらえるとうれしいです。

## evoto_restore.py（16bit 復元）

Evoto が実際に変えるのは画像の一部だけなので、レタッチされていない画素を元の 16bit データに戻します。

```bash
uv run evoto_restore.py 元の16bit.tif Evotoの出力.tif -o restored.tif --mask mask.tif
```

- 1 つ目の引数：Evoto に渡す前の 16bit TIFF
- 2 つ目の引数：Evoto の出力
- `-o`：出力ファイル。省略すると `<Evotoの出力>_restored16.tif`
- `--mask`：レタッチ判定のマスクを確認用に保存（省略可）

仕組みは次のとおりです。

1. 元データを Evoto と同じ規則で Adobe RGB/8bit に丸めて、「Evoto が何もしなかった場合の画像」を再現する
2. Evoto の出力と 2 段以上違う画素を「レタッチされた画素」としてマスクにする
3. マスクを少し広げてぼかし、レタッチされていない画素は元の 16bit データ、レタッチされた画素は Evoto の結果として合成する。出力は元データと同じ色空間/16bit

注意点です。

- Evoto に渡す前の 16bit TIFF を残しておいてください。入力と同じパスを `-o` や `--mask` に指定すると、上書きを防ぐためにエラーで止まります
- Evoto でトリミング、回転、リサイズをすると使えません（サイズが合わないとエラーで止まります）
- 全体に色や明るさをかけるプリセットを使うと、ほぼ全画素が「変化あり」になって効果がなくなります。変化した画素が 60% を超えると警告が出ます
- レタッチした部分そのものは、8bit 精度と Adobe RGB のままです
- 復元版には元データの ICC プロファイル、XMP、解像度が引き継がれますが、カメラの EXIF（機種名や撮影日時など）はコピーされません

## 制限

- 解析できるプロファイルは、マトリクス / TRC 型（sRGB、Adobe RGB、Display P3、ProPhoto など）だけです。LUT 型のプロファイルはエラーになります
- チャートに埋め込んだ自作の ICC プロファイルは、LittleCMS での変換結果と 8bit 換算 0.55 段以内で一致することを確認しています

## ライセンス

MIT License（`LICENSE` を参照）
