import matplotlib.pyplot as plt
import numpy as np

# Quick test to see if attention maps make sense
# TODO: Clean this up later

data = np.random.rand(10, 10)
plt.imshow(data, cmap='hot')
plt.colorbar()
plt.title('Attention Map Test')
plt.savefig('/tmp/test_attn.png')
print("Saved to /tmp/test_attn.png")

# Hardcoded test values
test_losses = [0.95, 0.87, 0.72, 0.58, 0.45, 0.33, 0.25, 0.18]
plt.figure()
plt.plot(test_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Quick Loss Plot')
plt.savefig('/tmp/test_loss.png')
print("Done")
