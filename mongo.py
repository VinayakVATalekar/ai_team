import pymongo


client=pymongo.MongoClient()

db=client["db"]

def order_store(dict):
    if client["db"]["order"].count_documents({})>=1:
        db["order"].drop()
    client["db"]["order"].insert_one(dict)
    return {"msg":"document added sucess"}

def drop_store(dict):
    if client["db"]["del"].count_documents({})>=1:
        db["del"].drop()
    client["db"]["del"].insert_one(dict)
    return {"msg":"document added sucess"}

def target_store(dict):
    if client["db"]["target"].count_documents({})>=1:
        db["target"].drop()
    client["db"]["target"].insert_one(dict)
    return {"msg":"document added sucess"}

def finding(value):
    res=client["db"][f"{value}"].find_one()
    return res
