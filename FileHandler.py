from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey


class FileHandler:
    """Class for working with files"""

    @staticmethod
    def read_file(file_path: str) -> bytes:
        """Reading a file"""
        try:
            with open(file_path, 'rb') as f:
                return f.read()
        except IOError as e:
            raise RuntimeError(f"Ошибка чтения файла {file_path}: {str(e)}")

    @staticmethod
    def write_file(data: bytes, file_path: str) -> None:
        """Write to file"""
        try:
            with open(file_path, 'wb') as f:
                f.write(data)
        except IOError as e:
            raise RuntimeError(f"Ошибка записи в файл {file_path}: {str(e)}")

    @staticmethod
    def serialize_keys(public_key: rsa.RSAPublicKey, public_key_path: str,
                       private_key: RSAPrivateKey, private_key_path: str) -> None:
        """Serializing keys to files"""
        try:
            with open(public_key_path, 'wb') as f:
                f.write(public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ))
            # print(f"Открытый ключ сохранен в {public_key_path}")

            with open(private_key_path, 'wb') as f:
                f.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            # print(f"Закрытый ключ сохранен в {private_key_path}")
        except IOError as e:
            raise RuntimeError(f"Ошибка записи ключей в файл: {str(e)}")
