由于部署服务器不支持连接huggingface，所以要把模型在有网络的机器上下载好，通过Dockerfile解压到docker容器内的~/.cache目录

zip打包过程
```shell
cd luxun
mkdir model

cd ~/.cache/huggingface/hub

zip -r /Users/zilliz/zilliz/luxun/model/models--BAAI--bge-small-zh-v1.5.zip models--BAAI--bge-small-zh-v1.5
```

unzip解压过程
```
unzip models--BAAI--bge-small-zh-v1.5.zip -d ~/.cache/huggingface/hub
```