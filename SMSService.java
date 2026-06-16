import java.util.ArrayList;
import java.util.List;

/**
 * A concrete implementation of {@link NotificationMedium} for sending SMS text messages.
 * * @author Judah Immanuel
 * @version 1.0
 */
public class SMSService implements NotificationMedium {
    /**
     * Simulates sending an SMS by printing to the standard output.
     * * @param message The text message to be dispatched.
     */
    @Override
    public void send(String message) {
        System.out.println("[SMSService] Sending SMS: " + message);
    }
}
