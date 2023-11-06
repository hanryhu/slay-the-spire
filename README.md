# Slay the Spire


### Wingboots

In Slay the Spire, the player climbs floors one at a time, visiting
one node per floor.  Each floor has a number of nodes which are
connected via edges to the nodes of the next floor, and ordinarily the
player can only traverse the edges present in the map.  Also, with the
Wingboots relic, a player can also visit an arbitrary node on the next
floor without edge constraints, using up a Charge of the Wingboots.
The wingboots come with 3 charges but not all of them must be
consumed.

Nodes can be elite nodes or non-elite nodes, but elite nodes contain
valuable rewards and the player may wish to optimize for them.  Given
that the wingboots have k charges (able to ignore edge constraints up
to k times), what is the optimum number of elites and how is it
achieved?

#### Usage:

Solve for the optimum number of elites in the act, where:

- The wingboots have `num_wingboots_charges` charges left.

- The elites in the act occur on floors `floor_number` (disambiguating
  between elites on the same floor using an optional `floor_id`).

- For each elite, `all_elites_reachable` lists descriptions for all
  future elites that can be reached without a wingboots charge.

```bash
python3 wingboots.py( [num_wingboots_charges])?( [elite_description])+

num_wingboots_charges:
    <int>

elite_description:
   <floor_number>(.<floor_id>)?(all_elites_reachable)

all_elites_reachable:
   (-<floor_number>(.<floor_id>)?)*
```

#### Examples:

-   Find the max elites that can be visited with no wingboots charges in
    a "double diamond" graph structure (fork in the path on floor 1 that
    merges at floor 2 and splits again at floor 3):

    ```
    python3 wingboots.py 0 1.0-2 1.1-2 2-3.0-3.1 3.0-4 3.1-4 4
    ```

    Note: The elite description `4` is optional in this case because it
    adds no new information.  However, no other elite description is
    optional because they include path information about other elites.

    <details><summary> Expected output: </summary>

    ```
    4 options found with 4 elites.
    Option 1:
        Floor 1, Node 0 (0 wingboots used)
        Floor 2, Node 0 (0 wingboots used)
        Floor 3, Node 0 (0 wingboots used)
        Floor 4, Node 0 (0 wingboots used)
    Option 2:
        Floor 1, Node 0 (0 wingboots used)
        Floor 2, Node 0 (0 wingboots used)
        Floor 3, Node 1 (0 wingboots used)
        Floor 4, Node 0 (0 wingboots used)
    Option 3:
        Floor 1, Node 1 (0 wingboots used)
        Floor 2, Node 0 (0 wingboots used)
        Floor 3, Node 0 (0 wingboots used)
        Floor 4, Node 0 (0 wingboots used)
    Option 4:
        Floor 1, Node 1 (0 wingboots used)
        Floor 2, Node 0 (0 wingboots used)
        Floor 3, Node 1 (0 wingboots used)
        Floor 4, Node 0 (0 wingboots used)
    ```

    </details>

-   Find the max elites that can be visited on Nov 6 daily challenge act
    1, with a hypothetical 3 wingboots charges:

    ```
    python3 wingboots.py 3 5.0-7-11-12 5.1-7-12 7-11
    ```

    screenshot of daily challenge act layout:

    <img width="269" alt="06 Nov 2023 daily challenge act layout" src="./img/06nov2023.png">

    <details><summary> Expected output: </summary>

    ```
    2 options found with 4 elites.
    Option 1:
        Floor 5, Node 0 (0 wingboots used)
        Floor 7, Node 0 (0 wingboots used)
        Floor 11, Node 0 (0 wingboots used)
        Jump prior to Floor 12, Node 0 (1 wingboots used)
    Option 2:
        Floor 5, Node 1 (0 wingboots used)
        Floor 7, Node 0 (0 wingboots used)
        Floor 11, Node 0 (0 wingboots used)
        Jump prior to Floor 12, Node 0 (1 wingboots used)
    ```

    </details>


#### Tests

Tests (`unittest.TestCase`) can be auto-collected by most Python test
runners such as nose2.

```
slay-the-spire$ python3 -m pip install nose2
slay-the-spire$ python3 -m nose2 -v wingboots
```

