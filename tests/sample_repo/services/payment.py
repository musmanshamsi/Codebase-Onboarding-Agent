"""Sample payment service module."""

def process_payment(order_id: str, amount: float) -> bool:
    print(f"Processing payment for order {order_id}: ${amount}")
    return charge_card(order_id, amount)

def charge_card(order_id: str, amount: float) -> bool:
    print(f"Charged card for {order_id}")
    return True

def refund_payment(transaction_id: str) -> bool:
    print(f"Refunding transaction {transaction_id}")
    return True
