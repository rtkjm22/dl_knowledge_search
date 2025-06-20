# 社内ナレッジ検索サイト

## 概要

本システムは、社内ドキュメントやナレッジベースをローカル環境で稼働する軽量な LLM（ローカル大規模言語モデル）に取り込み、社内メンバーが効率的に情報を検索・活用できる検索サイトを構築したものです。

🎯 主な目的
社内に蓄積されたナレッジ（仕様書、議事録、FAQ、手順書など）を横断的に検索可能にすることで、情報収集の時間短縮と業務効率の向上を図ります。
社外に公開できない機密情報を含むため、完全にオフラインで動作するセキュアな構成としています。
リポジトリ内の文書はサンプルになります。

🧠 システム構成
LLM は一旦軽量モデルを使用（`all-MiniLM-L6-v2`）。

ナレッジは定期的に学習され、ベクトル検索を通じて関連性の高い情報を抽出。
ユーザーはインプットフォームから自然文で質問を入力するだけで、該当する社内ドキュメントの抜粋や要約が得られる設計。

🛡 セキュリティと運用
インターネット経由での通信を一切行わず、ローカルネットワーク内で完結するアーキテクチャ。
社内ルールや部署別ナレッジなど、アクセス制御を考慮した権限ベースの検索制限にも対応予定。

💡 今後の展望

- 速度改善
- チャット形式での対話型インターフェース対応
- ユーザーごとの検索履歴・お気に入り登録機能
- ドキュメントの自動分類・要約精度の向上

## 動作イメージ

https://github.com/user-attachments/assets/b4ca3902-d5de-45b5-8c3a-b5aedc9b8834

## 起動方法

### 1. 仮想環境の作成（初回のみ）

```bash
python3 -m venv venv
```

### 2. 仮想環境の有効化

Mac/Linux

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

### 3. 依存ライブラリのインストール（初回のみ）

```bash
pip install -r requirements.txt
```

### 4. アプリの起動

```bash
python3 ./main.py
```

### 5. ブラウザでアクセス

```
http://127.0.0.1:7860/
```
