from pymilvus import MilvusClient

if __name__ == '__main__':
    client = MilvusClient(uri='./luxun.db');
    res = client.has_collection('LuXunWorks')
    print(res)
