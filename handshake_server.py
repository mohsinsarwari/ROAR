import logging
from ROAR.utilities_module.utilities import get_ip
import qrcode
import cv2
import numpy as np
import socket
from ROAR.utilities_module.utilities import get_ip

import socket


def showIPUntilAck():
    img = np.array(qrcode.make(f"{get_ip()}").convert('RGB'))
    success = False
    addr = None

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        s.bind((get_ip(), 8008))
        s.settimeout(1)
        while True:
            try:
                s.listen()

                cv2.imshow("Scan this code to connect to phone", img)
                k = cv2.waitKey(1) & 0xff
                if k == ord('q') or k == 27:
                    s.close()
                    break
                conn, addr = s.accept()
                addr = addr[0]
                if conn:
                    s.close()
                    success = True
                    break
            except socket.timeout as e:
                logging.info(f"Please tap on the ip address to scan QR code. ({get_ip()}:{8008}). {e}")
    except Exception as e:
        logging.error(f"Unable to bind socket: {e}")
    finally:
        s.close()
        cv2.destroyWindow("Scan this code to connect to phone")
    return success, addr


print(showIPUntilAck())
