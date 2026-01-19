import argparse
from HybridEncryptionSystem import *
from config import load_config


def main():
    parser = argparse.ArgumentParser(description='Гибридная система шифрования (IDEA + RSA)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-g', '--generate-keys', action='store_true', help='Генерация ключей')
    group.add_argument('-e', '--encrypt', action='store_true', help='Шифрование файла')
    group.add_argument('-d', '--decrypt', action='store_true', help='Дешифрование файла')

    parser.add_argument("-env", "--environment", choices=["dev", "prod"],
                        default=None, help="configuration environment (dev|prod)")
    parser.add_argument('-c', '--config_dir', default='config',
                       help='Директория с конфигурационными файлами')

    args = parser.parse_args()

    cfg = load_config(args.config_dir,args.environment)
    paths = cfg.paths
    dbg_cfg = cfg.debug

    if dbg_cfg.enabled and dbg_cfg.verbose:
        print(f"Инициализирована система шифрования: {cfg.app.name}")
        print(f"Окружение: {cfg.environment}")

    try:
        if args.generate_keys:
            HybridEncryptionSystem.generate_keys(
                paths.encrypted_key, paths.public_key, paths.private_key, dbg_cfg
            )
        elif args.encrypt:
            HybridEncryptionSystem.encrypt_file(
                paths.input_text, paths.private_key, paths.encrypted_key, paths.encrypted_text, dbg_cfg
            )
        elif args.decrypt:
            HybridEncryptionSystem.decrypt_file(
                paths.encrypted_text, paths.private_key, paths.encrypted_key, paths.decrypted_text, dbg_cfg
            )
    except Exception as e:
        print(f"Ошибка: {str(e)}")


if __name__ == "__main__":
    main()
