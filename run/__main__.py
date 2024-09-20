import sys

VALID_COMMANDS = {
    "train": ".train",
    "inference": ".inference",
    "scoring": ".scoring"
}

def print_initial_usage():
    print(
        f'[-] Only one command required which is in ("train", "inference", "interactive", "train_tokenizer")',
        file=sys.stderr,
    )

def print_command_usage():
    print(f'[-] Please type command in ("train", "inference")', file=sys.stderr)

def main():
    if len(sys.argv) < 2:
        print_initial_usage()
        return -1

    _, command, *arguments = sys.argv

    if command not in VALID_COMMANDS:
        print_command_usage()
        return -1

    module = VALID_COMMANDS[command]
    exec(f"from {module} import main, parser")
    return eval("main(parser.parse_args(arguments))")

if __name__ == "__main__":
    sys.exit(main())
    