import secrets
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding as sym_padding


class SymmetricEncryption:
    """Class for working with symmetric encryption (IDEA)"""

    @staticmethod
    def generate_key() -> bytes:
        """Generating a symmetric key"""
        try:
            symmetric_key = secrets.token_bytes(16)
            return symmetric_key
        except Exception as e:
            raise RuntimeError(f"Ошибка генерации симметричного ключа: {str(e)}")

    @staticmethod
    def add_padding(plaintext: bytes) -> bytes:
        """Adding padding to data"""
        try:
            padder = sym_padding.PKCS7(64).padder()
            padded_data = padder.update(plaintext) + padder.finalize()
            return padded_data
        except Exception as e:
            raise RuntimeError(f"Ошибка добавления padding: {str(e)}")

    @staticmethod
    def remove_padding(padded_data: bytes) -> bytes:
        """Removing padding from data"""
        try:
            unpadder = sym_padding.PKCS7(64).unpadder()
            data = unpadder.update(padded_data) + unpadder.finalize()
            return data
        except Exception as e:
            raise RuntimeError(f"Ошибка удаления padding: {str(e)}")

    @staticmethod
    def encrypt(padded_data: bytes, symmetric_key: bytes, iv: bytes) -> bytes:
        """IDEA Data Encryption"""
        try:
            cipher = Cipher(
                algorithms.IDEA(symmetric_key),
                modes.CBC(iv),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(padded_data) + encryptor.finalize()
            return ciphertext
        except Exception as e:
            raise RuntimeError(f"Ошибка шифрования IDEA: {str(e)}")

    @staticmethod
    def decrypt(encrypted_data: bytes, symmetric_key: bytes) -> bytes:
        """IDEA Data Decryption"""
        try:
            iv = encrypted_data[:8]
            ciphertext = encrypted_data[8:]
            cipher = Cipher(
                algorithms.IDEA(symmetric_key),
                modes.CBC(iv),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            decrypted_data = decryptor.update(ciphertext) + decryptor.finalize()
            return decrypted_data
        except Exception as e:
            raise RuntimeError(f"Ошибка дешифрования IDEA: {str(e)}")
