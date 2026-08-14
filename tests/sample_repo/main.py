"""Sample main entry point."""
from services.payment import process_payment

def main():
    print("Starting checkout flow...")
    process_payment("ORDER-123", 99.95)

if __name__ == "__main__":
    main()
