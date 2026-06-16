/** 
 * The underlying Java code was written by me without the assistance of GenAI. 
 * The reason for this was because I wanted more practice with Java, as I mainly use C#, which is Java-esque
 * However, I did use GenAI to generate comments and Javadoc tags. I queried the AI with the following query:
 * "Write descriptive Javadoc comments for the following code: [pasted code]. Do not alter or edit the underlying code."
 * I made some edits to the GenAI output, chief of which were the example notifications which were strangely specific, referencing random topics
 * For the WhatsAppService addition, I also didn't use GenAI because it was a copy/paste of the Email/SMS notification class, as deskgned. 
*/
import java.util.ArrayList;
import java.util.List;


/**
 * The main class to execute the program via CLI.
 * * @author Judah Immanuel
 * @version 1.0
 */
public class FlexibleNotificationSystem {
    
    /**
     * Default constructor for FlexibleNotificationSystem.
     * Explicitly defined to satisfy Javadoc requirements and prevent 
     * warnings regarding undocumented default constructors.
     */
    public FlexibleNotificationSystem() {
        // Empty constructor
    }
    
    /**
     * The main method demonstrating the use of Composition to swap notification
     * behaviors at runtime, and verifying the internal logging mechanism.
     * * @param args Command line arguments (not used).
     */
    public static void main(String[] args) {
        System.out.println("--- Starting Alert System ---");

        // 1. Initialize with EmailService
        NotificationMedium email = new EmailService();
        AlertSystem alertSystem = new AlertSystem(email);
        
        // Send a few alerts
        alertSystem.notifyUser("Email notification example 1");
        alertSystem.notifyUser("Email notification example 2");

        // 2. Swap the behavior at runtime to SMSService
        NotificationMedium sms = new SMSService();
        alertSystem.setMedium(sms);

        // Send a few more alerts
        alertSystem.notifyUser("SMS notification example 1");
        alertSystem.notifyUser("SMS notification example 2");

        // 3. Switch to WhatsApp service
        NotificationMedium whatsapp = new WhatsAppService(); 
        alertSystem.setMedium(whatsapp);

        // Send whatsapp alerts
        alertSystem.notifyUser("WhatsApp notification example 1");
        alertSystem.notifyUser("WhatsApp notification example 2");

        // 4. Display the session log (Using Collection Framework)
        System.out.println("\n--- Session Message Log ---");
        List<String> log = alertSystem.getMessageLog();
        for (int i = 0; i < log.size(); i++) {
            System.out.println((i + 1) + ". " + log.get(i));
        }
    }
}

