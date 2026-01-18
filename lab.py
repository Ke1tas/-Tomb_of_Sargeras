import argparse
from HybridEncryptionSystem import *


def main():
    parser = argparse.ArgumentParser(description='Гибридная система шифрования (IDEA + RSA)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-g', '--generate-keys', action='store_true', help='Генерация ключей')
    group.add_argument('-e', '--encrypt', action='store_true', help='Шифрование файла')
    group.add_argument('-d', '--decrypt', action='store_true', help='Дешифрование файла')

    parser.add_argument('-pr-k', '--private-key', help='Путь к закрытому ключу')
    parser.add_argument('-e-k', '--encrypted-key', help='Путь к зашифрованному симметричному ключу')
    parser.add_argument('-pub-k', '--public-key', help='Путь к открытому ключу (только для генерации)')
    parser.add_argument('-in-f', '--input-file', help='Путь к входному файлу')
    parser.add_argument('-out-f', '--output-file', help='Путь к выходному файлу')

    args = parser.parse_args()

    try:
        if args.generate_keys:
            if not all([args.encrypted_key, args.public_key, args.private_key]):
                parser.error("Для генерации ключей требуются --encrypted-key, --public-key и --private-key")
            HybridEncryptionSystem.generate_keys(
                args.encrypted_key, args.public_key, args.private_key
            )
        elif args.encrypt:
            if not all([args.input_file, args.private_key, args.encrypted_key, args.output_file]):
                parser.error("Для шифрования требуются --input-file, --private-key, --encrypted-key и --output-file")
            HybridEncryptionSystem.encrypt_file(
                args.input_file, args.private_key, args.encrypted_key, args.output_file
            )
        elif args.decrypt:
            if not all([args.input_file, args.private_key, args.encrypted_key, args.output_file]):
                parser.error("Для дешифрования требуются --input-file, --private-key, --encrypted-key и --output-file")
            HybridEncryptionSystem.decrypt_file(
                args.input_file, args.private_key, args.encrypted_key, args.output_file
            )
    except Exception as e:
        print(f"Ошибка: {str(e)}")


if __name__ == "__main__":
    main()
