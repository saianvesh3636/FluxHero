# Page snapshot

```yaml
- generic [active] [ref=e1]:
  - generic [ref=e6] [cursor=pointer]:
    - button "Open Next.js Dev Tools" [ref=e7]:
      - img [ref=e8]
    - generic [ref=e11]:
      - button "Open issues overlay" [ref=e12]:
        - generic [ref=e13]:
          - generic [ref=e14]: "0"
          - generic [ref=e15]: "1"
        - generic [ref=e16]: Issue
      - button "Collapse issues badge" [ref=e17]:
        - img [ref=e18]
  - alert [ref=e20]
  - generic [ref=e21]:
    - heading "Something went wrong" [level=2] [ref=e22]
    - paragraph [ref=e23]: An unexpected error occurred. The error has been logged for investigation.
    - group [ref=e24]:
      - generic "Error Details" [ref=e25] [cursor=pointer]
    - button "Try Again" [ref=e26] [cursor=pointer]
```