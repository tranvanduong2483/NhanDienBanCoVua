import json, requests


def getNextMove(FEN):
    URL = 'https://nextchessmove.com/api/v4/calculate'
    # Payload
    payload = {"kind": "remote", "fen": FEN, "position": {"fen": FEN, "moves": []}, "movetime": 5, "multipv": 1,
               "hardware": {"usePaidCpu": False, "usePaidGpu": False}, "engine": "sf11", "syzygy": False,
               "contempt": 24, "uuid": "d501108a-9366-4dc8-bebe-b1f375f4f51d"}
    # POST payload to nextchessmove.com
    print(" Gửi mã FEN lên nextchessmove.com")
    r = requests.post(URL, json=payload)
    # Response, status code
    response_json = json.loads(r.text)
    if r.status_code == 200:
        return response_json["move"]
    else:
        return -1


# Mã hóa FEN của ví trị con cờ trên bàn cờ
FEN = "r1bqk2r/pp1p1ppp/4p1n1/P1pb4/1P1P2P1/5N2/P2KPP1P/RNBQ1B1R w kq - 0 1"

FEN = input(" Mã FEN đầu vào: ")

move = getNextMove(FEN)
print(" Nước cờ tốt: ", move)
