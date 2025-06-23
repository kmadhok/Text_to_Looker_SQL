#!/usr/bin/env python3
"""
Generate synthetic retail data (star schema) and corresponding LookML metadata.
Creates 5 CSV files and fake_metadata.json that mimics Looker's API structure.
"""

import json
import random
from datetime import datetime, timedelta
from faker import Faker
import pandas as pd

# Initialize faker with seed for reproducibility
fake = Faker()
Faker.seed(42)
random.seed(42)

# Configuration
PROJECT_ID = "retail_sandbox"
DATASET_ID = "retail_sandbox"

def generate_customers(n=5000):
    """Generate customers table with US states and signup dates."""
    customers = []
    us_states = [
        'CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI',
        'NJ', 'VA', 'WA', 'AZ', 'MA', 'TN', 'IN', 'MO', 'MD', 'WI'
    ]
    
    for i in range(1, n + 1):
        customers.append({
            'customer_id': i,
            'first_name': fake.first_name(),
            'last_name': fake.last_name(),
            'email': fake.email(),
            'state': random.choice(us_states),
            'signup_date': fake.date_between(start_date='-2y', end_date='today').strftime('%Y-%m-%d'),
            'age': random.randint(18, 80)
        })
    
    return pd.DataFrame(customers)

def generate_products(n=1500):
    """Generate products table with categories and unit prices."""
    products = []
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books', 'Beauty', 'Toys']
    
    for i in range(1, n + 1):
        category = random.choice(categories)
        # Different price ranges by category
        if category == 'Electronics':
            price = round(random.uniform(50, 2000), 2)
        elif category == 'Clothing':
            price = round(random.uniform(20, 300), 2)
        elif category == 'Home & Garden':
            price = round(random.uniform(15, 500), 2)
        else:
            price = round(random.uniform(10, 200), 2)
            
        products.append({
            'product_id': i,
            'product_name': fake.catch_phrase(),
            'category': category,
            'unit_price': price,
            'brand': fake.company()
        })
    
    return pd.DataFrame(products)

def generate_stores(n=500):
    """Generate stores table with US cities."""
    stores = []
    cities = [
        'New York, NY', 'Los Angeles, CA', 'Chicago, IL', 'Houston, TX', 'Phoenix, AZ',
        'Philadelphia, PA', 'San Antonio, TX', 'San Diego, CA', 'Dallas, TX', 'San Jose, CA',
        'Austin, TX', 'Jacksonville, FL', 'Fort Worth, TX', 'Columbus, OH', 'Charlotte, NC',
        'San Francisco, CA', 'Indianapolis, IN', 'Seattle, WA', 'Denver, CO', 'Boston, MA'
    ]
    
    for i in range(1, n + 1):
        stores.append({
            'store_id': i,
            'store_name': f"{fake.company()} Store",
            'city': random.choice(cities),
            'region': random.choice(['North', 'South', 'East', 'West', 'Central']),
            'opened_date': fake.date_between(start_date='-5y', end_date='-1y').strftime('%Y-%m-%d')
        })
    
    return pd.DataFrame(stores)

def generate_orders(n=20000, customers_df=None, stores_df=None):
    """Generate orders table with FKs to customers and stores."""
    orders = []
    customer_ids = customers_df['customer_id'].tolist()
    store_ids = stores_df['store_id'].tolist()
    
    # Generate orders from today-90 days to today
    end_date = datetime.today()
    start_date = end_date - timedelta(days=90)
    
    for i in range(1, n + 1):
        order_date = fake.date_between(start_date=start_date, end_date=end_date)
        orders.append({
            'order_id': i,
            'customer_id': random.choice(customer_ids),
            'store_id': random.choice(store_ids),
            'order_date': order_date.strftime('%Y-%m-%d'),
            'order_status': random.choice(['completed', 'pending', 'cancelled', 'returned'])
        })
    
    return pd.DataFrame(orders)

def generate_order_items(n=80000, orders_df=None, products_df=None):
    """Generate order_items table with FKs to orders and products."""
    order_items = []
    order_ids = orders_df['order_id'].tolist()
    
    # Create product lookup for prices
    product_prices = dict(zip(products_df['product_id'], products_df['unit_price']))
    product_ids = products_df['product_id'].tolist()
    
    for i in range(1, n + 1):
        product_id = random.choice(product_ids)
        base_price = product_prices[product_id]
        # Add some price variation (discount/markup)
        unit_price = round(base_price * random.uniform(0.8, 1.2), 2)
        
        order_items.append({
            'order_item_id': i,
            'order_id': random.choice(order_ids),
            'product_id': product_id,
            'qty': random.randint(1, 5),
            'unit_price': unit_price,
            'line_total': round(unit_price * random.randint(1, 5), 2)
        })
    
    return pd.DataFrame(order_items)

def create_fake_metadata():
    """Create fake_metadata.json that mimics Looker's LookML structure."""
    
    # Helper function to create field objects
    def create_field(name, label, category, field_type, sql, is_hidden=False):
        return {
            "name": name,
            "label": label,
            "category": category,
            "type": field_type,
            "sql": sql,
            "is_hidden": is_hidden
        }
    
    # Define explores with their fields
    explores = [
        {
            "id": "customers",
            "name": "customers", 
            "label": "Customers",
            "sql_table_name": f"`{PROJECT_ID}.{DATASET_ID}.customers`",
            "fields": {
                "dimensions": [
                    create_field("customer_id", "Customer ID", "dimension", "number", "${TABLE}.customer_id"),
                    create_field("first_name", "First Name", "dimension", "string", "${TABLE}.first_name"),
                    create_field("last_name", "Last Name", "dimension", "string", "${TABLE}.last_name"),
                    create_field("email", "Email", "dimension", "string", "${TABLE}.email"),
                    create_field("state", "State", "dimension", "string", "${TABLE}.state"),
                    create_field("signup_date", "Signup Date", "dimension", "date", "${TABLE}.signup_date"),
                    create_field("age", "Age", "dimension", "number", "${TABLE}.age")
                ],
                "measures": [
                    create_field("count", "Count", "measure", "count", "*"),
                    create_field("avg_age", "Average Age", "measure", "average", "${TABLE}.age")
                ]
            }
        },
        {
            "id": "products",
            "name": "products",
            "label": "Products", 
            "sql_table_name": f"`{PROJECT_ID}.{DATASET_ID}.products`",
            "fields": {
                "dimensions": [
                    create_field("product_id", "Product ID", "dimension", "number", "${TABLE}.product_id"),
                    create_field("product_name", "Product Name", "dimension", "string", "${TABLE}.product_name"),
                    create_field("category", "Category", "dimension", "string", "${TABLE}.category"),
                    create_field("unit_price", "Unit Price", "dimension", "number", "${TABLE}.unit_price"),
                    create_field("brand", "Brand", "dimension", "string", "${TABLE}.brand")
                ],
                "measures": [
                    create_field("count", "Count", "measure", "count", "*"),
                    create_field("avg_price", "Average Price", "measure", "average", "${TABLE}.unit_price")
                ]
            }
        },
        {
            "id": "stores", 
            "name": "stores",
            "label": "Stores",
            "sql_table_name": f"`{PROJECT_ID}.{DATASET_ID}.stores`",
            "fields": {
                "dimensions": [
                    create_field("store_id", "Store ID", "dimension", "number", "${TABLE}.store_id"),
                    create_field("store_name", "Store Name", "dimension", "string", "${TABLE}.store_name"),
                    create_field("city", "City", "dimension", "string", "${TABLE}.city"),
                    create_field("region", "Region", "dimension", "string", "${TABLE}.region"),
                    create_field("opened_date", "Opened Date", "dimension", "date", "${TABLE}.opened_date")
                ],
                "measures": [
                    create_field("count", "Count", "measure", "count", "*")
                ]
            }
        },
        {
            "id": "orders",
            "name": "orders", 
            "label": "Orders",
            "sql_table_name": f"`{PROJECT_ID}.{DATASET_ID}.orders`",
            "fields": {
                "dimensions": [
                    create_field("order_id", "Order ID", "dimension", "number", "${TABLE}.order_id"),
                    create_field("customer_id", "Customer ID", "dimension", "number", "${TABLE}.customer_id"),
                    create_field("store_id", "Store ID", "dimension", "number", "${TABLE}.store_id"),
                    create_field("order_date", "Order Date", "dimension", "date", "${TABLE}.order_date"),
                    create_field("order_status", "Order Status", "dimension", "string", "${TABLE}.order_status")
                ],
                "measures": [
                    create_field("count", "Count", "measure", "count", "*")
                ]
            },
            "joins": [
                {
                    "name": "customers",
                    "type": "left_outer",
                    "relationship": "many_to_one",
                    "sql_on": "${orders.customer_id} = ${customers.customer_id} ;;"
                },
                {
                    "name": "stores",
                    "type": "left_outer",
                    "relationship": "many_to_one",
                    "sql_on": "${orders.store_id} = ${stores.store_id} ;;"
                },
                {
                    "name": "order_items",
                    "type": "left_outer",
                    "relationship": "one_to_many",
                    "sql_on": "${orders.order_id} = ${order_items.order_id} ;;"
                }
            ]
        },
        {
            "id": "order_items",
            "name": "order_items",
            "label": "Order Items", 
            "sql_table_name": f"`{PROJECT_ID}.{DATASET_ID}.order_items`",
            "fields": {
                "dimensions": [
                    create_field("order_item_id", "Order Item ID", "dimension", "number", "${TABLE}.order_item_id"),
                    create_field("order_id", "Order ID", "dimension", "number", "${TABLE}.order_id"),
                    create_field("product_id", "Product ID", "dimension", "number", "${TABLE}.product_id"),
                    create_field("qty", "Quantity", "dimension", "number", "${TABLE}.qty"),
                    create_field("unit_price", "Unit Price", "dimension", "number", "${TABLE}.unit_price"),
                    create_field("line_total", "Line Total", "dimension", "number", "${TABLE}.line_total")
                ],
                "measures": [
                    create_field("count", "Count", "measure", "count", "*"),
                    create_field("total_quantity", "Total Quantity", "measure", "sum", "${TABLE}.qty"),
                    create_field("avg_item_price", "Average Item Price", "measure", "average", "${TABLE}.unit_price"),
                    create_field("total_sales", "Total Sales", "measure", "sum", "${TABLE}.line_total")
                ]
            },
            "joins": [
                {
                    "name": "orders",
                    "type": "left_outer",
                    "relationship": "many_to_one",
                    "sql_on": "${order_items.order_id} = ${orders.order_id} ;;"
                },
                {
                    "name": "products",
                    "type": "left_outer",
                    "relationship": "many_to_one",
                    "sql_on": "${order_items.product_id} = ${products.product_id} ;;"
                }
            ]
        }
    ]
    
    metadata = {
        "model": "retail",
        "explores": explores
    }
    
    return metadata

def main():
    """Generate all synthetic data and metadata."""
    print("🔄 Generating synthetic retail data...")
    
    # Generate tables
    print("   📊 Generating customers...")
    customers_df = generate_customers(5000)
    customers_df.to_csv('data/customers.csv', index=False)
    
    print("   📦 Generating products...")
    products_df = generate_products(1500) 
    products_df.to_csv('data/products.csv', index=False)
    
    print("   🏪 Generating stores...")
    stores_df = generate_stores(500)
    stores_df.to_csv('data/stores.csv', index=False)
    
    print("   🛒 Generating orders...")
    orders_df = generate_orders(20000, customers_df, stores_df)
    orders_df.to_csv('data/orders.csv', index=False)
    
    print("   📝 Generating order items...")
    order_items_df = generate_order_items(80000, orders_df, products_df)
    order_items_df.to_csv('data/order_items.csv', index=False)
    
    # Generate metadata
    print("   📋 Generating fake metadata...")
    metadata = create_fake_metadata()
    with open('fake_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    total_size = (
        customers_df.memory_usage(deep=True).sum() +
        products_df.memory_usage(deep=True).sum() +
        stores_df.memory_usage(deep=True).sum() +
        orders_df.memory_usage(deep=True).sum() +
        order_items_df.memory_usage(deep=True).sum()
    ) / (1024 * 1024)  # Convert to MB
    
    print(f"\n✅ Data generation complete!")
    print(f"   📁 Generated 5 CSV files in data/ directory")
    print(f"   📄 Generated fake_metadata.json")
    print(f"   💾 Total memory footprint: ~{total_size:.1f} MB")
    print(f"   📊 Row counts:")
    print(f"      - customers: {len(customers_df):,}")
    print(f"      - products: {len(products_df):,}")
    print(f"      - stores: {len(stores_df):,}")
    print(f"      - orders: {len(orders_df):,}")
    print(f"      - order_items: {len(order_items_df):,}")

if __name__ == "__main__":
    main() 