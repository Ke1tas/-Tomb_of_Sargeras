from AsymmetricEncryption import *
from SymmetricEncryption import *
from FileHandler import *


class HybridEncryptionSystem:
    """Main class for managing hybrid encryption"""

    @staticmethod
    def generate_keys(encrypted_key_path: str, public_key_path: str,
                      private_key_path: str) -> None:
        """Generate all keys"""
        print("Генерация ключей...")
        try:
            symmetric_key = SymmetricEncryption.generate_key()
            private_key, public_key = AsymmetricEncryption.generate_keys()

            AsymmetricEncryption.serialize_keys(
                public_key, public_key_path,
                private_key, private_key_path
            )
            AsymmetricEncryption.encrypt_symmetric_key(
                symmetric_key, public_key,
                encrypted_key_path
            )
            print("Генерация ключей завершена успешно!")
        except Exception as e:
            raise RuntimeError(f"Ошибка в процессе генерации ключей: {str(e)}")

    @staticmethod
    def encrypt_file(input_file_path: str, private_key_path: str,
                     encrypted_key_path: str, output_file_path: str) -> None:
        """File encryption"""
        print(f"Шифрование файла {input_file_path}...")
        try:
            private_key = AsymmetricEncryption.load_private_key(private_key_path)
            encrypted_symmetric_key = FileHandler.read_file(encrypted_key_path)
            symmetric_key = AsymmetricEncryption.decrypt_symmetric_key(
                encrypted_symmetric_key, private_key
            )

            plaintext = FileHandler.read_file(input_file_path)
            padded_data = SymmetricEncryption.add_padding(plaintext)
            iv = secrets.token_bytes(8)
            ciphertext = SymmetricEncryption.encrypt(padded_data, symmetric_key, iv)

            FileHandler.write_file(iv + ciphertext, output_file_path)
            print(f"Файл успешно зашифрован и сохранен в {output_file_path}")
        except Exception as e:
            raise RuntimeError(f"Ошибка шифрования файла: {str(e)}")

    @staticmethod
    def decrypt_file(input_file_path: str, private_key_path: str,
                     encrypted_key_path: str, output_file_path: str) -> None:
        """File decryption"""
        print(f"Дешифрование файла {input_file_path}...")
        try:
            private_key = AsymmetricEncryption.load_private_key(private_key_path)
            encrypted_symmetric_key = FileHandler.read_file(encrypted_key_path)
            symmetric_key = AsymmetricEncryption.decrypt_symmetric_key(
                encrypted_symmetric_key, private_key
            )

            encrypted_text = FileHandler.read_file(input_file_path)
            decrypted_padded_data = SymmetricEncryption.decrypt(
                encrypted_text, symmetric_key
            )
            decrypted_data = SymmetricEncryption.remove_padding(decrypted_padded_data)

            FileHandler.write_file(decrypted_data, output_file_path)
            print(f"Файл успешно расшифрован и сохранен в {output_file_path}")
        except Exception as e:
            raise RuntimeError(f"Ошибка дешифрования файла: {str(e)}")
