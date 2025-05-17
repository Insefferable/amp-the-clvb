#!/usr/bin/env python3
"""
nightclub simulation
- simulates customer drinking behavior and service interactions
- tracks blood alcohol content (BAC) and gimmick states
- models queuing system with bartenders and waiters
- records performance metrics per hour
"""

import random
import itertools
import heapq
import csv
from collections import deque

# --- core parameters ---
# shot and drink specifications
SHOT_VOLUME_ML = 44.0        # standard shot size
BRANDY_ABV = 0.45           # alcohol by volume (45%)
MIN_SHOT_VOLUME_ML = 40.0   # min shot

# BAC calculation factors
ETHANOL_DENSITY_G_PER_ML = 0.789
BLOOD_VOLUME_L_PER_KG = 0.067
DEFAULT_WEIGHT_KG = 70.0    # average customer weight
FEMALE_BAC_MULTIPLIER = 1.25 # Badat (2023) BAC approx for women, based on water ratio

# Detting (2007) BAC Elimination Rates (per minute); although diff is non significant
ELIM_RATE_MALE_MEAN_PER_HOUR = 0.0159
ELIM_RATE_MALE_STD_PER_HOUR = 0.0028
ELIM_RATE_FEMALE_MEAN_PER_HOUR = 0.0179
ELIM_RATE_FEMALE_STD_PER_HOUR = 0.0030

# service flow controls
BASE_RATE = 0.1             # base order generation rate
P_DIRECT_WALK = 0.3         # chance to go directly to bar
INTERJECTION_CHANCE = 0.1   # chance for other tables to join order

# gimmick (e.g. chat, dance, etc.) and Eduardo et. al. (2024) gridlock settings
GIMMICK_CHANCE = 0.3        # post-drink gimmick probability
GIMMICK_DURATION = 5.0      # minutes in gimmick state
GRIDLOCK_THRESHOLD = 0.4    # % of gimmick customers to trigger gridlock
GRIDLOCK_DURATION = 10.0    # minutes of gridlock

# --- event types ---
event_counter = itertools.count()  # unique event id generator
ORDER_REQUEST = 'order_request'    # new order being placed
TAKE_ORDER = 'take_order'         # waiter taking order
bartender_SERVE = 'bartender_serve'   # bottle obtained
CONSUME_DONE = 'consume_done'     # customer finished drinking
ELIMINATE_BAC = 'eliminate_bac'   # bac reduction cycle

class Customer:
    """models individual customer behavior and state"""
    _ids = itertools.count()  # unique customer id counter

    def __init__(self, gender, table):
        self.id = next(Customer._ids)
        self.gender = gender
        # generate random weight around mean
        self.weight = random.expovariate(1/DEFAULT_WEIGHT_KG)
        self.bac = 0.0
        self.table = table
        
        # set gender-specific factors
        self.gender_mult = FEMALE_BAC_MULTIPLIER if gender == 'F' else 1.0

        # Sample elimination rate from normal distribution (per hour), then convert to per minute
        if gender == 'F':
            elim_rate_hr = max(0, random.gauss(ELIM_RATE_FEMALE_MEAN_PER_HOUR, ELIM_RATE_FEMALE_STD_PER_HOUR))
        else:
            elim_rate_hr = max(0, random.gauss(ELIM_RATE_MALE_MEAN_PER_HOUR, ELIM_RATE_MALE_STD_PER_HOUR))
        self.elim_rate = elim_rate_hr / 60.0

        # calculate BAC increase per drink based on weight
        dL = self.weight * BLOOD_VOLUME_L_PER_KG * 10
        self.bac_inc = (
            SHOT_VOLUME_ML * BRANDY_ABV * ETHANOL_DENSITY_G_PER_ML / dL
        ) * self.gender_mult

        # gimmick state tracking
        self.in_gimmick = False
        self.gimmick_end_time = 0.0

    def eliminate(self):
        """reduce the customer's bac over time"""
        self.bac = max(0, self.bac - self.elim_rate)

    def check_gimmick_state(self, now):
        """update and return customer's gimmick state"""
        if self.in_gimmick and now >= self.gimmick_end_time:
            self.in_gimmick = False
        return self.in_gimmick

    def start_gimmick(self, now):
        """start a gimmick period for this customer"""
        if random.random() < GIMMICK_CHANCE:
            self.in_gimmick = True
            self.gimmick_end_time = now + GIMMICK_DURATION


class Table:
    """
    represents a table in the nightclub
    manages customer seating and drink ordering
    """
    _ids = itertools.count()

    def __init__(self, capacity, stop_threshold=0.2):
        self.id = next(Table._ids)
        self.capacity = capacity
        self.customers = []
        self.bottle_ml = 750.0  # standard bottle size
        self.pending = False
        self.stop_threshold = stop_threshold
        self.recent = deque()  # recent order history
        self.cooldown = 0.0


class Server:
    """
    represents a server (bartender or waiter)
    tracks availability and idle status
    """
    def __init__(self):
        self.idle = True
        self.available = 0.0


# -----------------------------------------------------------------------------
# helper functions
# -----------------------------------------------------------------------------
def compute_consume_time(cust, mean_consume_min):
    """
    compute drink consumption time based on customer's current bac
    higher bac leads to slower drinking
    """
    base_rate = mean_consume_min * (1 + cust.bac / 0.5)
    return random.uniform(base_rate * 0.5, base_rate * 1.5)


def schedule_request(now, table, events, params):
    """
    schedule a new order request for a table
    handles cooldown periods and bac limits
    """
    # check if table can place orders
    if table.pending:
        #print(f"Warning: schedule_request called for Table {table.id} which is already pending!")
        return
    if now < table.cooldown:
        return
        
    # check bac limits
    if all(cust.bac >= 0.5 for cust in table.customers):
        return
        
    # track recent orders
    table.recent.append(now)
    while table.recent and now - table.recent[0] > 5:
        table.recent.popleft()
        
    # apply cooldown if too many recent orders
    if len(table.recent) > 4:
        table.cooldown = now + 2.0
        
    # calculate order rate based on eligible customers
    rate = sum(cust.bac < 0.5 for cust in table.customers) * BASE_RATE
    if rate <= 0:
        return
        
    # schedule next order
    delay = random.expovariate(rate) * params['order_scale']
    if random.random() < params['gimmick']:
        delay += random.uniform(params['min'], params['max'])
        
    eid = next(event_counter)
    heapq.heappush(events, (now + delay, eid, ORDER_REQUEST, table))


# -----------------------------------------------------------------------------
# Main Simulation Function
# -----------------------------------------------------------------------------
def simulate_clvb(
    sim_time=300,  # 5 hours in minutes
    table_configs=[(20, 4), (7, 6), (1, 6)],
    bartender_count=4,
    waiter_count=7,
    mean_bartender_min=1.0,      # 1 minute service time
    mean_consume_min=15.0,     # 15 minutes to consume a drink
    reneger_th_min=3.0,
    reneger_prop=0.3,
    gimmick=0.3,
    min_g=2.0,
    max_g=6.0,
    order_scale=2.0
):
    """
    Main simulation function
    - Sets up nightclub environment
    - Runs event-driven simulation
    - Tracks metrics and generates reports
    """
    random.seed()

    # Initialize tables and customers
    tables = []
    customers = []
    for num, cap in table_configs:
        for _ in range(num):
            t = Table(cap)
            for _ in range(cap):
                c = Customer(random.choice(['M', 'F']), t)
                t.customers.append(c)
                customers.append(c)
            tables.append(t)

    # Initialize servers
    bartenders = [Server() for _ in range(bartender_count)]
    waiters = [Server() for _ in range(waiter_count)]

    # Event queue and counter
    events = []
    params = {'gimmick': gimmick, 'min': min_g, 'max': max_g, 'order_scale': order_scale}

    # CSV log
    log = csv.writer(open('order_log.csv', 'w', newline=''))
    log.writerow(['order_id', 'stage', 'time_sec', 'wait_sec', 'svc_sec', 'idle_sec', 'time_diff'])
    oid = 0
    stats = {
        'wait': [], 'svc': [], 'consume': [], 'reneged': 0, 'idle': [],
        'gridlocks': 0,  # Track number of gridlocks
        'current_gridlock': False,  # Track current gridlock state
        'gridlock_end': 0.0,  # Track when current gridlock ends
        'queue_wait_times': {  # Add queue wait time tracking
            'bartender': [],
            'waiter': []
        }
    }
    hourly = [{
        'orders': 0, 
        'reneged': 0, 
        'wait': [], 
        'svc': [], 
        'consume': [], 
        'idle': [],
        'bartender_count': 0,  # Add counters for each queue
        'waiter_count': 0,
        'queue_wait_bartender': [],  # Add queue wait tracking per hour
        'queue_wait_waiter': []
    } for _ in range(int(sim_time // 60) + 1)]

    # Schedule initial order requests
    now = 0.0
    for t in tables:
        schedule_request(0, t, events, params)

    pending = deque()
    next_hr = 60

    # Initialize a variable to track the time of the previous order
    previous_time_sec = None

    # Schedule the first BAC elimination event
    event_id = next(event_counter)
    heapq.heappush(events, (now + 1.0, event_id, ELIMINATE_BAC, None))

    # Main loop
    while events and now < sim_time:
        now, eid, etype, obj = heapq.heappop(events)

        # Hourly summary
        if now >= next_hr:
            hr_idx = int(next_hr // 60) - 1
            hd = hourly[hr_idx]
            n = hd['orders']
            
            # Calculate queue statistics
            bartender_waits = hd['queue_wait_bartender']
            waiter_waits = hd['queue_wait_waiter']
            
            avg_bartender_wait = (sum(bartender_waits) / len(bartender_waits) if bartender_waits else 0)
            max_bartender_wait = max(bartender_waits) if bartender_waits else 0
            
            avg_waiter_wait = (sum(waiter_waits) / len(waiter_waits) if waiter_waits else 0)
            max_waiter_wait = max(waiter_waits) if waiter_waits else 0
            
            print(f"=== Hour {hr_idx + 1} ===")
            print(f" Orders: {n}, Reneged: {hd['reneged']}")
            print(f" Avg Wait: {(sum(hd['wait']) / n if n else 0):.2f} min")
            print(f" Avg Service: {(sum(hd['svc']) / n if n else 0):.2f} min")
            print(f" Avg Idle: {(sum(hd['idle']) / len(hd['idle']) if hd['idle'] else 0):.2f} min")
            print(f" bartender Queue: {hd['bartender_count']}")  # Use accumulated count
            print(f" Waiter Queue: {hd['waiter_count']}")    # Use accumulated count
            print(f" Queue Statistics (bartender):")
            print(f"   Average Wait: {avg_bartender_wait:.2f} min")
            print(f"   Maximum Wait: {max_bartender_wait:.2f} min")
            print(f"   Orders in Queue: {len(bartender_waits)}")
            print(f" Queue Statistics (Waiter):")
            print(f"   Average Wait: {avg_waiter_wait:.2f} min")
            print(f"   Maximum Wait: {max_waiter_wait:.2f} min")
            print(f"   Orders in Queue: {len(waiter_waits)}")
            next_hr += 60

        # Handle events
        if etype == ORDER_REQUEST:
            tbl = obj
            if tbl.pending:
                #print(f"Warning: ORDER_REQUEST for Table {tbl.id} which is already pending!")
                continue  # Already has a pending order, skip
            tbl.pending = True
            dest = 'bartender' if random.random() < P_DIRECT_WALK else 'waiter'
            entry_time = now
            pending.append((entry_time, tbl, dest))
            hr_idx = int(now // 60)
            if dest == 'bartender':
                hourly[hr_idx]['bartender_count'] += 1
            else:
                hourly[hr_idx]['waiter_count'] += 1

        elif etype == TAKE_ORDER:
            srv, tbl, t0 = obj
            srv.idle = True
            srv.available = now

            # Collect interjection orders from nearby tables
            bulk_orders = [tbl]  # Start with the current table
            for other_tbl in tables:
                if other_tbl != tbl and not other_tbl.pending and random.random() < INTERJECTION_CHANCE:
                    other_tbl.pending = True
                    bulk_orders.append(other_tbl)
                #elif other_tbl != tbl and other_tbl.pending and random.random() < INTERJECTION_CHANCE:
                    #print(f"Warning: Interjection tried to add Table {other_tbl.id} which is already pending!")

            # Log all bulk orders as individual entries in the bartender queue
            for bulk_tbl in bulk_orders:
                if bulk_tbl.pending:
                    pending.append((now, bulk_tbl, 'bartender'))
                #else:
                    #print(f"Warning: bulk_tbl {bulk_tbl.id} not pending when adding to queue!")

        elif etype == bartender_SERVE:
            srv, tbl, t0 = obj
            srv.idle = True
            srv.available = now
            shot = random.triangular(MIN_SHOT_VOLUME_ML, SHOT_VOLUME_ML, SHOT_VOLUME_ML)
            used = min(shot, tbl.bottle_ml)
            tbl.bottle_ml -= used
            cust = random.choice(tbl.customers)

            # Enforce the 0.5 BAC rule
            if cust.bac < 0.5:
                cust.bac = min(0.5, cust.bac + cust.bac_inc)
                # Maybe start a gimmick after drinking
                cust.start_gimmick(now)
                ctime = compute_consume_time(cust, mean_consume_min)
                stats['consume'].append(ctime)
                hourly[int(now // 60)]['consume'].append(ctime)
                eid = next(event_counter)
                heapq.heappush(events, (now + ctime, eid, CONSUME_DONE, tbl))
            # else:
                # Comment out bartender pings
                # print(f"Customer {cust.id} at Table {tbl.id} has BAC >= 0.5 and stops drinking.")

        elif etype == CONSUME_DONE:
            tbl = obj
            tbl.pending = False
            
            # Check for potential gridlock after consumption
            gimmick_count = sum(1 for c in customers if c.check_gimmick_state(now))
            gimmick_ratio = gimmick_count / len(customers)
            
            if gimmick_ratio >= GRIDLOCK_THRESHOLD and not stats['current_gridlock']:
                stats['current_gridlock'] = True
                stats['gridlock_end'] = now + GRIDLOCK_DURATION
                stats['gridlocks'] += 1
                # During gridlock, delay new orders significantly
                delay = GRIDLOCK_DURATION * 0.5
            else:
                delay = 0.0
                
            eid = next(event_counter)
            heapq.heappush(events, (now + delay, eid, ORDER_REQUEST, tbl))

        elif etype == ELIMINATE_BAC:
            # Eliminate BAC for all customers
            for c in customers:
                c.eliminate()
            # Schedule the next BAC elimination event
            event_id = next(event_counter)
            heapq.heappush(events, (now + 1.0, event_id, ELIMINATE_BAC, None))  # Schedule for the next minute

        # Update waiter assignment to handle interjection orders
        for w in waiters:
            if w.idle and pending and pending[0][2] == 'waiter':
                entry_time, tbl, _ = pending.popleft()
                start = max(now, w.available)
                queue_wait = (start - entry_time)  # Time in queue before service
                
                # Log queue wait time
                stats['queue_wait_times']['waiter'].append(queue_wait)
                hr_idx = int(now // 60)
                hourly[hr_idx]['queue_wait_waiter'].append(queue_wait)
                
                wait_t = start - entry_time
                svc_min = random.expovariate(1 / mean_bartender_min)
                oid += 1
                log.writerow([
                    oid, 'waiter', f"{start * 60:.2f}", f"{wait_t * 60:.2f}",
                    f"{svc_min * 60:.2f}", f"{(now - w.available) * 60:.2f}", f"{(start - entry_time) * 60:.2f}"
                ])
                stats['wait'].append(wait_t)
                stats['svc'].append(svc_min)
                stats['idle'].append(now - w.available)
                hr_idx = int(start // 60)
                hourly[hr_idx]['wait'].append(wait_t)
                hourly[hr_idx]['svc'].append(svc_min)
                hourly[hr_idx]['idle'].append(now - w.available)
                hourly[hr_idx]['orders'] += 1
                w.idle = False
                finish = start + svc_min
                w.available = finish
                eid = next(event_counter)
                heapq.heappush(events, (finish, eid, TAKE_ORDER, (w, tbl, entry_time)))

        # Update bartender assignment to process orders individually
        for b in bartenders:
            if b.idle and pending and pending[0][2] == 'bartender':
                entry_time, tbl, _ = pending.popleft()
                start = max(now, b.available)
                queue_wait = (start - entry_time)  # Time in queue before service
                
                # Log queue wait time
                stats['queue_wait_times']['bartender'].append(queue_wait)
                hr_idx = int(now // 60)
                hourly[hr_idx]['queue_wait_bartender'].append(queue_wait)
                
                wait_t = start - entry_time
                if wait_t > reneger_th_min and random.random() < reneger_prop:
                    stats['reneged'] += 1
                    hourly[int(start // 60)]['reneged'] += 1
                    tbl.pending = False
                    eid = next(event_counter)
                    heapq.heappush(events, (start, eid, ORDER_REQUEST, tbl))
                    continue
                svc_min = random.expovariate(1 / mean_bartender_min)
                oid += 1
                log.writerow([
                    oid, 'bartender', f"{start * 60:.2f}", f"{wait_t * 60:.2f}",
                    f"{svc_min * 60:.2f}", f"{(now - b.available) * 60:.2f}", f"{(start - entry_time) * 60:.2f}"
                ])
                stats['wait'].append(wait_t)
                stats['svc'].append(svc_min)
                stats['idle'].append(now - b.available)
                hr_idx = int(start // 60)
                hourly[hr_idx]['wait'].append(wait_t)
                hourly[hr_idx]['svc'].append(svc_min)
                hourly[hr_idx]['idle'].append(now - b.available)
                hourly[hr_idx]['orders'] += 1
                b.idle = False
                finish = start + svc_min
                b.available = finish
                eid = next(event_counter)
                heapq.heappush(events, (finish, eid, bartender_SERVE, (b, tbl, entry_time)))

        # Add gridlock check to main loop
        if stats['current_gridlock'] and now >= stats['gridlock_end']:
            stats['current_gridlock'] = False

    print(f"Total Orders: {oid}, Reneged: {stats['reneged']}")
    print(f"Total Gridlocks: {stats['gridlocks']}")
    if stats['wait']:
        print(f"Avg Wait: {sum(stats['wait']) / len(stats['wait']):.2f} min")
    if stats['svc']:
        print(f"Avg Service: {sum(stats['svc']) / len(stats['svc']):.2f} min")
    if stats['idle']:
        print(f"Avg Idle: {sum(stats['idle']) / len(stats['idle']):.2f} min")
    if stats['queue_wait_times']['bartender']:
        print(f"Avg Queue Wait (bartender): {sum(stats['queue_wait_times']['bartender']) / len(stats['queue_wait_times']['bartender']):.2f} min")
    if stats['queue_wait_times']['waiter']:
        print(f"Avg Queue Wait (Waiter): {sum(stats['queue_wait_times']['waiter']) / len(stats['queue_wait_times']['waiter']):.2f} min")


if __name__ == '__main__':
    simulate_clvb()
