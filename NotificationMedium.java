import java.util.ArrayList;
import java.util.List;


public interface NotificationMedium {
    /**
     * Sends a notification message.
     * * @param message The text message to be sent to the user.
     */
    void send(String message);
}

