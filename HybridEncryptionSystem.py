from omegaconf import DictConfig

from AsymmetricEncryption import *
from SymmetricEncryption import *
from FileHandler import *


class HybridEncryptionSystem:
    """Main class for managing hybrid encryption"""

    @staticmethod
    def generate_keys(encrypted_key_path: str, public_key_path: str,
                      private_key_path: str, debug_config: DictConfig) -> None:
        """Generate all keys"""
        if debug_config.enabled:
            print("Генерация ключей...")
        try:
            symmetric_key = SymmetricEncryption.generate_key()
            private_key, public_key = AsymmetricEncryption.generate_keys()

            FileHandler.serialize_keys(
                public_key, public_key_path,
                private_key, private_key_path
            )
            AsymmetricEncryption.encrypt_symmetric_key(
                symmetric_key, public_key,
                encrypted_key_path
            )
            if debug_config.enabled:
                print("Генерация ключей завершена успешно!")

                if debug_config.verbose:
                    print(f"Ключи сохранены в:")
                    print(f"  - Открытый ключ: {public_key_path}")
                    print(f"  - Закрытый ключ: {private_key_path}")
                    print(f"  - Зашифрованный симметричный ключ: {encrypted_key_path}")

        except Exception as e:
            raise RuntimeError(f"Ошибка в процессе генерации ключей: {str(e)}")

    @staticmethod
    def encrypt_file(input_file_path: str, private_key_path: str,
                     encrypted_key_path: str, output_file_path: str, debug_config: DictConfig) -> None:
        """File encryption"""
        if debug_config.enabled:
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

            if debug_config.enabled:
                print(f"Файл успешно зашифрован и сохранен в {output_file_path}")
                if debug_config.verbose:
                    print(f"Размер исходного файла: {len(plaintext)} байт")
                    print(f"Размер зашифрованного файла: {len(ciphertext) + len(iv)} байт")


        except Exception as e:
            raise RuntimeError(f"Ошибка шифрования файла: {str(e)}")

    @staticmethod
    def decrypt_file(input_file_path: str, private_key_path: str,
                     encrypted_key_path: str, output_file_path: str, debug_config: DictConfig) -> None:
        """File decryption"""
        if debug_config.enabled:
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
            if debug_config.enabled:
                print(f"Файл успешно расшифрован и сохранен в {output_file_path}")
                if debug_config.verbose:
                    print(f"Размер зашифрованного файла: {len(encrypted_text)} байт")
                    print(f"Размер расшифрованного файла: {len(decrypted_data)} байт")
        except Exception as e:
            raise RuntimeError(f"Ошибка дешифрования файла: {str(e)}")
